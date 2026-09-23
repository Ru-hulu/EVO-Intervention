/* Local-only viewer. All measurements come from scores.csv, loaded via data.js. */
(() => {
  'use strict';
  const $ = id => document.getElementById(id);
  const DATA = window.ACC_DATA;
  if (!DATA?.episodes?.length) {
    $('error').hidden = false;
    $('error').textContent = '未找到评分数据，请先运行 build_viewer.py。';
    return;
  }
  const colors = ['#078575', '#d57721'];
  const method = DATA.method;
  const hz = Number((1 / method.queryIntervalSeconds).toFixed(4));
  const previous = method.fixedPreviousSelection;
  const counts = group => DATA.episodes.filter(e => e.group === group).length;
  document.querySelector('.dataset-note').textContent = `${counts('hf_success')} 条示范 · ${counts('local_failure')} 条失败轨迹 · ${hz} Hz · 无归一化（mm）`;
  document.querySelector('.model-note').textContent = `15k 模型 · H${method.actionHorizon} / R${method.executedSteps} · 视频 ${method.videoFPS} FPS`;
  $('sampling-note').textContent = `ACC ${hz} Hz，每 ${method.queryIntervalSeconds} 秒更新；视频 ${method.videoFPS} FPS，原速播放。游标连续移动，数值保持到下一次评分。`;
  $('parameter-note').textContent = `EMA α = ${method.emaAlpha} · 无归一化 · 仅比较接下来 6 步 · 单位 mm`;
  $('method-explanation').textContent = '本次复用相同的 15k 模型动作预测缓存，仅重新计算指标：上次 [6:12] 对本次 [0:6]，对 6×3 个 XYZ 坐标差取平均绝对值，再从米换算为毫米；不做速度/跨度归一化，不是欧氏距离。EMA α=0.9、初始为0，首个观测无评分。相机画面复用原视频中的相机区域，右侧曲线和数值为本次重算。执行窗口指计划执行的前6步，不是与真实执行轨迹的误差。HF 示范无原生成功标签；四条样本延用旧选样，本次不重新排名。此为自定义ACC变体，不是论文原始指标。';
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const number = v => v == null ? '—' : v.toFixed(3);
  const time = t => `${String(Math.floor(t / 60)).padStart(2, '0')}:${(t % 60).toFixed(1).padStart(4, '0')}`;
  const escape = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;', '<':'&lt;', '>':'&gt;', '"':'&quot;', "'":'&#39;'}[c]));
  const state = { episodes: [], videos: [], plots: [], master: null, start: 0, end: 20, duration: 0, current: 0, playing: false, token: 0, overview: null, drag: null };
  const modes = [
    {id: 'compare-failure', title: previous ? `失败组 · ${hz} Hz 重算对比` : '失败组 · 最好 vs 最坏', ids: DATA.selection.local_failure},
    {id: 'compare-success', title: previous ? `成功组示范 · ${hz} Hz 重算对比` : '成功组示范 · 最好 vs 最坏', ids: DATA.selection.hf_success},
    ...DATA.episodes.map(e => ({id: e.id, title: e.title + (previous ? ` · ${hz} Hz 重算` : ''), ids: [e.id]})),
  ];
  $('trajectory').innerHTML = modes.map(m => `<option value="${escape(m.id)}">${escape(m.title)}</option>`).join('');

  function message(text = '') {
    $('error').textContent = text;
    $('error').hidden = !text;
  }

  function pause() {
    state.playing = false;
    state.videos.forEach(v => v.pause());
    paintCursor();
  }

  function valueAt(e, values, t) {
    if (t < 0 || t >= e.duration) return null;
    let lo = 0, hi = e.times.length;
    while (lo < hi) {
      const mid = (lo + hi) >>> 1;
      if (e.times[mid] <= t + 1e-7) lo = mid + 1; else hi = mid;
    }
    return lo ? values[lo - 1] : null;
  }

  function columns(e) {
    return $('metric').value === 'raw' ? [e.rawLeft, e.rawRight] : [e.left, e.right];
  }

  function seekTo(t) {
    state.current = clamp(t, state.start, state.end);
    state.videos.forEach((v, i) => {
      const duration = Number.isFinite(v.duration) ? v.duration : state.episodes[i].duration;
      const target = clamp(state.current, 0, Math.max(0, duration - 1 / 30));
      if (v.readyState > 0) v.currentTime = target;
      if (state.current >= duration) v.pause();
      else if (state.playing && v.paused) v.play().catch(() => {});
    });
    paintCursor();
  }

  async function play() {
    message();
    if (state.current >= state.end - 0.035) seekTo(state.start);
    const token = state.token;
    state.playing = true;
    paintCursor();
    try {
      await Promise.all(state.videos.map((v, i) => state.current < state.episodes[i].duration ? v.play() : Promise.resolve()));
      if (token !== state.token || !state.playing) return;
      paintCursor();
    } catch (error) {
      if (token !== state.token || !state.playing) return;
      pause();
      if (error.name !== 'AbortError') message('视频暂时无法播放。请确认视频文件夹与网页目录位于同一结果目录中。');
    }
  }

  function updatePlayback() {
    if (!state.playing || !state.master) return;
    const t = state.master.currentTime;
    if (t >= state.end - 0.015 || state.master.ended) {
      if ($('loop').checked) {
        seekTo(state.start);
        play();
      } else {
        pause();
        seekTo(state.end);
      }
      return;
    }
    state.current = clamp(t, state.start, state.end);
    state.videos.forEach((v, i) => {
      if (v === state.master) return;
      if (t >= state.episodes[i].duration) {
        v.pause();
      } else if (!state.master.seeking && !v.seeking && v.readyState >= 2 && Math.abs(v.currentTime - t) > 0.12) {
        v.currentTime = t;
      }
    });
    paintCursor();
  }

  function frame() {
    updatePlayback();
    requestAnimationFrame(frame);
  }

  function applyCamera() {
    const crops = {
      base: [24, 134, 960, 540], left: [24, 720, 470, 270],
      right: [514, 720, 470, 270], all: [24, 134, 960, 856],
    };
    const [x, y, w, h] = crops[$('camera').value];
    state.videos.forEach(v => {
      v.parentElement.style.aspectRatio = `${w} / ${h}`;
      v.parentElement.style.setProperty('--camera-ratio', w / h);
      Object.assign(v.style, {width: `${1920 / w * 100}%`, left: `${-x / w * 100}%`, top: `${-y / h * 100}%`});
    });
  }

  function selectMode() {
    pause();
    state.token++;
    const mode = modes.find(m => m.id === $('trajectory').value);
    state.episodes = mode.ids.map(id => DATA.episodes.find(e => e.id === id));
    state.duration = Math.max(...state.episodes.map(e => e.duration));
    state.start = 0;
    state.end = Math.min(20, state.duration);
    state.current = 0;
    state.plots = [];
    const pair = state.episodes.length === 2;
    document.body.classList.toggle('single', !pair);
    $('videos').classList.toggle('pair', pair);
    $('charts').classList.toggle('charts-pair', pair);
    const name = (e, i) => previous ? `${e.title} · ${hz} Hz 重算` : pair ? `${i === 0 ? '最好 · ACC 最低' : '最坏 · ACC 最高'} · #${e.episodeNumber}` : e.title;
    $('videos').innerHTML = state.episodes.map((e, i) => `
      <div class="video-entry"><div class="video-caption"><span>${escape(name(e, i))}</span><small>${previous ? '本次' : '全程'} P95 ${e.p95.toFixed(3)} mm · ${time(e.duration)}</small></div>
      <div class="viewport"><video muted playsinline preload="auto" src="${escape(e.src)}" aria-label="${escape(e.title)}"></video>
      <span class="ended-label" hidden>本条轨迹已结束</span><span class="media-error" hidden>视频未能加载，请检查本地文件路径。</span></div></div>`).join('');
    $('charts').innerHTML = state.episodes.map((e, i) => `<div class="chart-row"><div class="chart-row-header"><span class="chart-name">${escape(name(e, i))}</span><span class="live-values"><span class="live-left">左 —</span><span class="live-right">右 —</span><span class="live-ended" hidden>已结束</span></span></div><svg class="plot" role="img" aria-label="${escape(e.title)} ACC 曲线，点击跳转播放时间"></svg></div>`).join('');
    state.videos = [...$('videos').querySelectorAll('video')];
    const masterIndex = state.episodes.findIndex(e => e.duration === state.duration);
    state.master = state.videos[masterIndex];
    state.videos.forEach((v, i) => {
      v.muted = true;
      v.playbackRate = Number($('rate').value);
      v.addEventListener('loadedmetadata', () => {
        v.currentTime = Math.min(state.current, Math.max(0, v.duration - 1 / 30));
      });
      v.addEventListener('error', () => { v.parentElement.querySelector('.media-error').hidden = false; });
      if (i === masterIndex) {
        v.addEventListener('timeupdate', updatePlayback);
        v.addEventListener('ended', updatePlayback);
      }
    });
    [...$('charts').querySelectorAll('.plot')].forEach(svg => svg.addEventListener('click', event => {
      const plot = state.plots.find(p => p.svg === svg);
      if (!plot) return;
      const rect = svg.getBoundingClientRect();
      const x = (event.clientX - rect.left) * plot.width / rect.width;
      seekTo(state.start + clamp((x - plot.left) / plot.innerWidth, 0, 1) * (state.end - state.start));
    }));
    applyCamera();
    updateRangeUI();
    draw();
    message();
  }

  function updateRangeUI() {
    $('start').value = state.start.toFixed(2);
    $('end').value = state.end.toFixed(2);
    $('start').max = state.duration;
    $('end').max = state.duration;
    $('seek').min = state.start;
    $('seek').max = state.end;
    $('range-caption').textContent = `${state.start.toFixed(1)} — ${state.end.toFixed(1)} 秒 · 共 ${(state.end - state.start).toFixed(1)} 秒`;
    $('clip-start').textContent = time(state.start);
    $('clip-end').textContent = time(state.end);
  }

  function setRange(start, end, move = true) {
    if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end > state.duration + 0.01 || end - start < 0.5) {
      message(`请输入 0 到 ${state.duration.toFixed(2)} 秒之间的区间，长度至少 0.5 秒。`);
      return false;
    }
    pause();
    message();
    state.start = start;
    state.end = Math.min(end, state.duration);
    state.current = clamp(state.current, state.start, state.end);
    updateRangeUI();
    draw();
    if (move) seekTo(start);
    return true;
  }

  function zoom(factor) {
    const span = clamp((state.end - state.start) * factor, 0.5, state.duration);
    const center = (state.start + state.end) / 2;
    const start = clamp(center - span / 2, 0, state.duration - span);
    setRange(start, start + span);
  }

  // Hold each sampled value until the next observation. No interpolated scores.
  function pathFor(e, values, start, end, x, y) {
    let path = '', previousEnd = -Infinity;
    for (let i = 0; i < e.times.length; i++) {
      const a = Math.max(start, e.times[i]);
      const b = Math.min(end, e.times[i + 1] ?? e.duration, e.duration);
      if (values[i] == null || b <= a) continue;
      const command = Math.abs(a - previousEnd) < 1e-7 ? 'L' : 'M';
      path += `${command}${x(a).toFixed(2)},${y(values[i]).toFixed(2)}H${x(b).toFixed(2)}`;
      previousEnd = b;
    }
    const endpoint = valueAt(e, values, end);
    if (endpoint != null && Math.abs(previousEnd - end) < 1e-7) path += `L${x(end).toFixed(2)},${y(endpoint).toFixed(2)}`;
    return path;
  }

  function yDomain() {
    const values = [];
    DATA.episodes.forEach(e => columns(e).forEach(series => series.forEach((v, i) => {
      if (v != null && e.times[i] <= state.end && (e.times[i + 1] ?? e.duration) > state.start) values.push(v);
    })));
    if (!values.length) return [0, 1];
    let lo = Math.min(...values), hi = Math.max(...values);
    const pad = Math.max((hi - lo) * 0.1, hi * 0.01, 0.005);
    lo = $('zero').checked ? 0 : Math.max(0, lo - pad);
    hi += pad;
    return [lo, hi];
  }

  function drawPlots() {
    const [lo, hi] = yDomain();
    state.plots = [...$('charts').querySelectorAll('.plot')].map((svg, index) => {
      const e = state.episodes[index], rect = svg.getBoundingClientRect();
      const width = Math.max(290, rect.width), height = rect.height || 240;
      const left = 48, right = 14, top = 28, bottom = 34;
      const innerWidth = width - left - right, innerHeight = height - top - bottom;
      const x = t => left + (t - state.start) / (state.end - state.start) * innerWidth;
      const y = v => top + (hi - v) / (hi - lo) * innerHeight;
      svg.setAttribute('viewBox', `0 0 ${width} ${height}`);
      Object.assign(svg.dataset, {start: state.start, end: state.end, yMin: lo, yMax: hi, episode: e.id});
      let html = '';
      for (let j = 0; j <= 4; j++) {
        const v = lo + (hi - lo) * j / 4, yy = y(v);
        html += `<line class="grid" x1="${left}" y1="${yy}" x2="${width - right}" y2="${yy}"/><text class="axis-label" x="${left - 7}" y="${yy + 3}" text-anchor="end">${v < 1 ? v.toFixed(3) : v.toFixed(2)}</text>`;
      }
      const ticks = width < 400 ? 4 : 5;
      for (let j = 0; j <= ticks; j++) {
        const t = state.start + (state.end - state.start) * j / ticks;
        html += `<line class="grid" x1="${x(t)}" y1="${top}" x2="${x(t)}" y2="${height - bottom}"/><text class="axis-label" x="${x(t)}" y="${height - bottom + 19}" text-anchor="${j === ticks ? 'end' : 'middle'}">${t.toFixed(1)}s</text>`;
      }
      if (e.duration < state.end) {
        const xx = x(Math.max(e.duration, state.start));
        html += `<rect x="${xx}" y="${top}" width="${Math.max(0, width - right - xx)}" height="${innerHeight}" fill="#f2f4f7"/><text class="axis-label" x="${Math.min(xx + 5, width - 55)}" y="${top + 13}">已结束</text>`;
      }
      const series = columns(e);
      series.forEach((values, i) => { html += `<path class="trace" stroke="${colors[i]}" d="${pathFor(e, values, state.start, state.end, x, y)}"/>`; });
      if (!series.some(v => pathFor(e, v, state.start, state.end, x, y))) html += `<text class="axis-label" x="${width / 2}" y="${height / 2}" text-anchor="middle">此区间暂无有效 ACC（首次推理无前一动作块）</text>`;
      html += `<line class="playhead" y1="${top - 5}" y2="${height - bottom}"/><text class="plot-time" y="16"></text><circle class="point" r="4" fill="${colors[0]}"/><circle class="point" r="4" fill="${colors[1]}"/>`;
      svg.innerHTML = html;
      return {svg, e, series, width, left, innerWidth, x, y, line: svg.querySelector('.playhead'), label: svg.querySelector('.plot-time'), points: [...svg.querySelectorAll('.point')]};
    });
    $('scale-note').textContent = `四条共用纵轴 ${lo.toFixed(2)}–${hi.toFixed(2)} mm`;
  }

  function drawOverview() {
    const svg = $('overview'), width = Math.max(280, svg.getBoundingClientRect().width);
    const left = 18, right = 18, top = 9, bottom = 57;
    const x = t => left + t / state.duration * (width - left - right);
    const max = Math.max(0.01, ...state.episodes.flatMap(e => e.max.filter(v => v != null))) * 1.1;
    const y = v => bottom - v / max * (bottom - top);
    svg.setAttribute('viewBox', `0 0 ${width} 82`);
    let html = `<rect x="${left}" y="${top}" width="${width - left - right}" height="${bottom - top}" fill="#f5f8fa"/>`;
    state.episodes.forEach((e, i) => { html += `<path d="${pathFor(e, e.max, 0, state.duration, x, y)}" fill="none" stroke="${i ? '#8593a8' : '#82b9b0'}" stroke-width="1.2"/>`; });
    html += `<rect x="${x(state.start)}" y="${top}" width="${x(state.end) - x(state.start)}" height="${bottom - top}" fill="#078575" fill-opacity=".1" stroke="#078575"/>`;
    for (const [edge, t] of [['start', state.start], ['end', state.end]]) {
      html += `<g class="handle" data-edge="${edge}"><rect x="${x(t) - 7}" y="${top - 3}" width="14" height="${bottom - top + 6}" fill="transparent"/><rect x="${x(t) - 2.5}" y="${top - 1}" width="5" height="${bottom - top + 2}" rx="2" fill="#078575"/></g>`;
    }
    for (let j = 0; j <= 5; j++) {
      const t = state.duration * j / 5;
      html += `<text class="axis-label" x="${x(t)}" y="75" text-anchor="${j === 0 ? 'start' : j === 5 ? 'end' : 'middle'}">${time(t)}</text>`;
    }
    html += `<line class="overview-cursor" stroke="#1a2a3b" y1="${top - 2}" y2="${bottom + 2}" pointer-events="none"/>`;
    svg.innerHTML = html;
    state.overview = {width, left, right, x, line: svg.querySelector('.overview-cursor')};
  }

  function draw() {
    drawPlots();
    drawOverview();
    paintCursor();
  }

  function paintCursor() {
    const t = state.current;
    $('current-time').textContent = time(t);
    $('seek').value = t;
    $('clip-progress').textContent = `片段 ${(t - state.start).toFixed(1)} / ${(state.end - state.start).toFixed(1)} 秒`;
    $('play').textContent = state.playing ? 'Ⅱ 暂停' : '▶ 播放片段';
    $('play').setAttribute('aria-label', state.playing ? '暂停' : '播放片段');
    $('play-status').textContent = state.playing ? '正在播放选定片段' : t >= state.end - 0.02 ? '片段已结束' : '已暂停 · 可点击曲线定位';
    state.plots.forEach((p, i) => {
      const xx = p.x(t);
      p.svg.dataset.current = t;
      p.line.setAttribute('x1', xx);
      p.line.setAttribute('x2', xx);
      p.label.setAttribute('x', clamp(xx, p.left + 20, p.width - 35));
      p.label.setAttribute('text-anchor', 'middle');
      p.label.textContent = `${t.toFixed(2)} s`;
      const values = p.series.map(v => valueAt(p.e, v, t));
      p.points.forEach((point, k) => {
        point.style.display = values[k] == null ? 'none' : '';
        if (values[k] != null) { point.setAttribute('cx', xx); point.setAttribute('cy', p.y(values[k])); }
      });
      const row = p.svg.parentElement;
      row.querySelector('.live-left').textContent = `左 ${number(values[0])} mm`;
      row.querySelector('.live-right').textContent = `右 ${number(values[1])} mm`;
      row.querySelector('.live-ended').hidden = t < p.e.duration;
      state.videos[i].parentElement.querySelector('.ended-label').hidden = t < p.e.duration;
    });
    if (state.overview) {
      state.overview.line.setAttribute('x1', state.overview.x(t));
      state.overview.line.setAttribute('x2', state.overview.x(t));
    }
  }

  function overviewTime(event) {
    const rect = $('overview').getBoundingClientRect(), p = state.overview;
    return clamp(((event.clientX - rect.left) * p.width / rect.width - p.left) / (p.width - p.left - p.right) * state.duration, 0, state.duration);
  }
  $('overview').addEventListener('pointerdown', event => {
    if (event.button !== 0) return;
    pause();
    const anchor = overviewTime(event);
    state.drag = {anchor, x: event.clientX, edge: event.target.closest('[data-edge]')?.dataset.edge, start: state.start, end: state.end, moved: false};
    $('overview').setPointerCapture(event.pointerId);
    event.preventDefault();
  });
  $('overview').addEventListener('pointermove', event => {
    const d = state.drag;
    if (!d || (!d.moved && Math.abs(event.clientX - d.x) < 3)) return;
    d.moved = true;
    const t = overviewTime(event);
    let a, b;
    if (d.edge === 'start') { a = Math.min(t, d.end - 0.5); b = d.end; }
    else if (d.edge === 'end') { a = d.start; b = Math.max(t, d.start + 0.5); }
    else { a = Math.min(d.anchor, t); b = Math.max(d.anchor, t); if (b - a < 0.5) return; }
    setRange(a, b, false);
  });
  function endDrag(event) {
    const d = state.drag;
    if (!d) return;
    state.drag = null;
    if ($('overview').hasPointerCapture(event.pointerId)) $('overview').releasePointerCapture(event.pointerId);
    if (d.moved) seekTo(state.start);
    else seekTo(d.anchor);
  }
  $('overview').addEventListener('pointerup', endDrag);
  $('overview').addEventListener('pointercancel', endDrag);
  $('trajectory').addEventListener('change', selectMode);
  $('camera').addEventListener('change', applyCamera);
  $('apply').addEventListener('click', () => setRange(Number($('start').value), Number($('end').value)));
  ['start', 'end'].forEach(id => $(id).addEventListener('keydown', event => { if (event.key === 'Enter') $('apply').click(); }));
  $('zoom-in').addEventListener('click', () => zoom(0.5));
  $('zoom-out').addEventListener('click', () => zoom(2));
  $('full').addEventListener('click', () => setRange(0, state.duration));
  $('play').addEventListener('click', () => state.playing ? pause() : play());
  $('restart').addEventListener('click', () => { seekTo(state.start); play(); });
  $('seek').addEventListener('input', () => seekTo(Number($('seek').value)));
  $('rate').addEventListener('change', () => state.videos.forEach(v => { v.playbackRate = Number($('rate').value); }));
  $('metric').addEventListener('change', draw);
  $('zero').addEventListener('change', draw);
  document.addEventListener('keydown', event => {
    if (event.code === 'Space' && !['INPUT', 'SELECT', 'BUTTON', 'TEXTAREA'].includes(event.target.tagName)) { event.preventDefault(); state.playing ? pause() : play(); }
  });
  let resizeTimer;
  window.addEventListener('resize', () => { clearTimeout(resizeTimer); resizeTimer = setTimeout(draw, 100); });
  selectMode();
  requestAnimationFrame(frame);
})();

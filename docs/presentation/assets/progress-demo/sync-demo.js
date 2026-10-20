(() => {
  'use strict';
  const data = window.progressDemoData;
  const video = document.getElementById('opening-progress-video');
  if (!data || !video || !data.frames.length) return;
  const el = id => document.getElementById(id);
  const chart = el('opening-progress-chart');
  const play = el('opening-play');
  const seek = el('opening-seek');
  const clock = el('opening-clock');
  const status = el('opening-demo-status');
  const reveal = el('opening-reveal-rect');
  const cursor = el('opening-chart-cursor');
  const truthDot = el('opening-truth-dot');
  const predDot = el('opening-prediction-dot');
  const truthValue = el('opening-truth-value');
  const predValue = el('opening-prediction-value');
  const left = 64, right = 880, top = 16, bottom = 226;
  const x = t => left + (right - left) * t / data.duration;
  const y = p => bottom - (bottom - top) * p;
  const clampTime = t => Math.max(0, Math.min(Number.isFinite(t) ? t : 0, data.duration));
  const svg = (tag, attrs, label) => {
    const node = document.createElementNS('http://www.w3.org/2000/svg', tag);
    Object.entries(attrs).forEach(([name, value]) => node.setAttribute(name, value));
    if (label !== undefined) node.textContent = label;
    return node;
  };
  const grid = el('opening-chart-grid');
  [0, .25, .5, .75, 1].forEach(p => {
    grid.append(svg('line', {x1:left, x2:right, y1:y(p), y2:y(p), stroke:'#dfe5ec'}));
    grid.append(svg('text', {x:left-12, y:y(p)+6, 'text-anchor':'end'}, `${p*100}%`));
  });
  [0, 5, 10, 15, 20, 25, 30].forEach(t => {
    grid.append(svg('text', {x:x(t), y:254, 'text-anchor':t === 30 ? 'end' : 'middle'}, t === 30 ? '30 s' : String(t)));
  });
  const path = column => data.frames.map((f, i) => `${i ? 'L' : 'M'}${x(f[0]).toFixed(3)},${y(f[column]).toFixed(3)}`).join(' ');
  el('opening-truth-path').setAttribute('d', path(1));
  el('opening-prediction-path').setAttribute('d', path(2));
  seek.max = data.duration;
  seek.step = 1/data.fps;
  video.controls = false;

  function frameAt(time) {
    let lo = 0, hi = data.frames.length - 1;
    while (lo < hi) {
      const mid = Math.ceil((lo + hi) / 2);
      if (data.frames[mid][0] <= time + 1e-6) lo = mid;
      else hi = mid - 1;
    }
    return lo;
  }
  function render(time) {
    const t = clampTime(time);
    const index = frameAt(t);
    const row = data.frames[index];
    // The path, dots and numbers all use the same saved frame, including seeks.
    const at = x(row[0]);
    reveal.setAttribute('width', Math.max(0, at-left));
    cursor.setAttribute('x1', at); cursor.setAttribute('x2', at);
    truthDot.setAttribute('cx', at); truthDot.setAttribute('cy', y(row[1]));
    predDot.setAttribute('cx', at); predDot.setAttribute('cy', y(row[2]));
    truthValue.textContent = `${(row[1]*100).toFixed(1)}%`;
    predValue.textContent = `${(row[2]*100).toFixed(1)}%`;
    seek.value = t;
    clock.textContent = `${t.toFixed(1)} / ${data.duration.toFixed(1)} s`;
    seek.setAttribute('aria-valuetext', `${t.toFixed(1)} 秒，预测进度 ${(row[2]*100).toFixed(1)}%`);
    chart.dataset.frame = index;
    chart.dataset.mediaTime = t;
  }
  let active = false, started = false, callback = null, callbackKind = null;
  function cancelFrame() {
    if (callback !== null) {
      if (callbackKind === 'video') video.cancelVideoFrameCallback(callback);
      else cancelAnimationFrame(callback);
    }
    callback = null;
  }
  function scheduleFrame() {
    cancelFrame();
    if (!active || video.paused || video.ended || document.hidden) return;
    if (typeof video.requestVideoFrameCallback === 'function') {
      callbackKind = 'video';
      callback = video.requestVideoFrameCallback((_, metadata) => {
        callback = null;
        render(metadata.mediaTime);
        scheduleFrame();
      });
    } else {
      callbackKind = 'animation';
      callback = requestAnimationFrame(() => {
        callback = null;
        render(video.currentTime);
        scheduleFrame();
      });
    }
  }
  function updatePlayButton() {
    play.textContent = video.ended ? '重播' : video.paused ? '播放' : '暂停';
    play.setAttribute('aria-label', `${play.textContent}同步演示`);
  }
  async function start() {
    if (video.ended) video.currentTime = 0;
    try { await video.play(); }
    catch { updatePlayButton(); }
  }
  play.addEventListener('click', () => video.paused ? start() : video.pause());
  seek.addEventListener('input', () => {
    video.currentTime = clampTime(Number(seek.value));
    render(video.currentTime);
  });
  el('opening-speed').addEventListener('change', event => { video.playbackRate = Number(event.target.value); });
  chart.addEventListener('click', event => {
    const box = chart.getBoundingClientRect();
    const clickedX = (event.clientX - box.left) * 900 / box.width;
    video.currentTime = clampTime((clickedX-left)/(right-left)*data.duration);
    render(video.currentTime);
  });
  video.addEventListener('play', () => { updatePlayButton(); scheduleFrame(); });
  ['pause', 'ended'].forEach(name => video.addEventListener(name, () => {
    cancelFrame(); render(video.currentTime); updatePlayButton();
  }));
  ['loadedmetadata', 'seeked'].forEach(name => video.addEventListener(name, () => {
    render(video.currentTime); updatePlayButton();
  }));
  video.addEventListener('timeupdate', () => {
    if (video.paused || typeof video.requestVideoFrameCallback !== 'function') render(video.currentTime);
  });
  video.addEventListener('error', () => {
    status.textContent = '视频加载失败，请检查同目录下的视频文件';
    play.disabled = true;
  });
  document.addEventListener('visibilitychange', () => {
    if (document.hidden) video.pause();
  });
  window.progressDemo = {
    setActive(value) {
      active = value;
      if (!active) { video.pause(); cancelFrame(); return; }
      render(video.currentTime);
      if (!started) {
        started = true;
        if (!window.matchMedia('(prefers-reduced-motion: reduce)').matches) start();
      }
    }
  };
  render(0);
})();

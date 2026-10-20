(() => {
  const data = window.astraInterventionData;
  const video = document.getElementById('astra-rollout');
  if (!data || !video) return;
  const label = document.getElementById('astra-controller');
  const play = document.getElementById('astra-play');
  const jump = document.getElementById('astra-jump');
  const time = document.getElementById('astra-time');
  const timeline = document.getElementById('astra-timeline');
  const head = document.getElementById('astra-playhead');
  const interventions = data.segments.filter(s => s.mode === 'astra');
  const first = interventions[0];
  let nextIntervention = 0;
  let clipEnd = null;
  let animation;
  data.segments.forEach(segment => {
    const block = document.createElement('span');
    block.className = 'as-segment'; block.dataset.mode = segment.mode;
    block.style.width = `${(segment.end - segment.start) / data.control_duration * 100}%`;
    block.title = `${segment.mode === 'astra' ? 'Astra 修正' : 'π₀.₅ 策略执行'} ${segment.start.toFixed(2)}–${segment.end.toFixed(2)} s`;
    timeline.insertBefore(block, head);
    if (segment.mode === 'astra') {
      const marker = document.createElement('span');
      marker.className = 'as-intervention-marker';
      marker.style.left = `${segment.start / data.control_duration * 100}%`;
      marker.setAttribute('aria-hidden', 'true');
      timeline.insertBefore(marker, head);
    }
  });
  function update() {
    const t = video.currentTime;
    const segment = data.segments.find(s => t >= s.start && t < s.end) || data.segments[data.segments.length - 1];
    label.dataset.mode = segment.mode;
    label.textContent = t >= data.control_duration ? '回放结束' : segment.mode === 'astra' ? `Astra 介入 · 第 ${interventions.indexOf(segment) + 1} / ${interventions.length} 段` : 'π₀.₅ 策略执行';
    head.style.left = `${Math.min(t / data.control_duration, 1) * 100}%`;
    timeline.setAttribute('aria-valuenow', String(Math.min(t, data.control_duration)));
    timeline.setAttribute('aria-valuetext', `${t.toFixed(1)} 秒，${label.textContent}`);
    time.textContent = `${t.toFixed(1)} / ${data.video_duration.toFixed(1)} s${video.playbackRate === .25 ? ' · ¼×' : ''}`;
    play.textContent = video.paused ? '播放全程' : '暂停';
    if (clipEnd !== null && t >= clipEnd && !video.paused) video.pause();
  }
  function animate() { update(); if (!video.paused) animation = requestAnimationFrame(animate); }
  play.addEventListener('click', async () => {
    if (!video.paused) video.pause();
    else {
      clipEnd = null; video.playbackRate = 1; video.currentTime = 0;
      try { await video.play(); } catch { play.textContent = '点击重试'; }
    }
  });
  jump.addEventListener('click', async () => {
    if (!interventions.length) return;
    const selected = interventions[nextIntervention];
    nextIntervention = (nextIntervention + 1) % interventions.length;
    clipEnd = Math.min(selected.end + .25, data.video_duration);
    video.currentTime = Math.max(selected.start - .2, 0);
    video.playbackRate = .25;
    try { await video.play(); } catch { play.textContent = '点击重试'; }
  });
  function seek(t) {
    video.pause(); clipEnd = null; video.playbackRate = 1;
    video.currentTime = Math.min(Math.max(t, 0), data.control_duration);
  }
  const initialize = () => seek(first ? first.start + .2 : 0);
  if (video.readyState >= 1) initialize();
  else video.addEventListener('loadedmetadata', initialize, { once:true });
  video.addEventListener('play', () => { cancelAnimationFrame(animation); animate(); });
  video.addEventListener('pause', () => { cancelAnimationFrame(animation); update(); });
  ['loadedmetadata','timeupdate','seeked','ended','ratechange'].forEach(event => video.addEventListener(event, update));
  timeline.addEventListener('click', event => {
    const box = timeline.getBoundingClientRect();
    seek((event.clientX - box.left) / box.width * data.control_duration);
  });
  timeline.addEventListener('keydown', event => {
    if (!['ArrowLeft','ArrowRight','Home','End'].includes(event.key)) return;
    event.preventDefault(); event.stopPropagation();
    seek(event.key === 'Home' ? 0 : event.key === 'End' ? data.control_duration : video.currentTime + (event.key === 'ArrowLeft' ? -.2 : .2));
  });
})();

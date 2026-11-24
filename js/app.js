(function () {
  'use strict';
  const PRE = 96, POST = 144, FS = 360;
  const CL = BeatModel.CLASSES;
  const COL = { N: '#17695f', S: '#b36b00', V: '#c4301e', F: '#5b3fa0' };
  const NAMES = { N: 'Normal', S: 'Supraventricular', V: 'Ventricular', F: 'Fusion' };
  const $ = (id) => document.getElementById(id);
  const pct = (v, d) => (v * 100).toFixed(d == null ? 1 : d) + '%';
  const clip = (v, a, b) => Math.min(b, Math.max(a, v));

  let MODEL = null, METRICS = null, RECS = null, cur = null, playing = true, speed = 1, showTruth = false, t = 0, last = 0, selected = null;
  const reduceMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;

  function windowOf(rec, b) { return Float32Array.from(rec._raw[b.k]); }
  function classify(rec, b) {
    if (!b.probs) {
      b.probs = BeatModel.predict(MODEL, rec, b.k);
      let k = 0; for (let i = 1; i < 4; i++) if (b.probs[i] > b.probs[k]) k = i;
      b.pred = k;
    }
    return b;
  }

  // ---------- strip ----------
  const cv = $('strip'), ctx = cv.getContext('2d');
  let W = 0, H = 0, dpr = 1;
  const PX_PER_SEC = 110, PX_PER_MV = 120;
  function resize() {
    dpr = Math.min(2, devicePixelRatio || 1);
    W = cv.clientWidth; H = cv.clientHeight;
    cv.width = W * dpr; cv.height = H * dpr; ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }
  function chip(x, y, label, color, ring) {
    ctx.font = '500 12px "Plex Mono", monospace';
    const w = 22, h = 20;
    ctx.fillStyle = color; ctx.beginPath(); ctx.roundRect(x - w / 2, y - h, w, h, 3); ctx.fill();
    ctx.fillStyle = '#fff'; ctx.textAlign = 'center'; ctx.fillText(label, x, y - 6);
    if (ring) { ctx.strokeStyle = ring; ctx.lineWidth = 2; ctx.beginPath(); ctx.roundRect(x - w / 2 - 2, y - h - 2, w + 4, h + 4, 5); ctx.stroke(); }
  }
  function draw() {
    ctx.clearRect(0, 0, W, H);
    if (!cur) return;
    const sig = cur.sig, mid = H * 0.62, edge = W - 28;
    const nowS = t * FS;                       // sample index at right edge
    const spp = FS / PX_PER_SEC;               // samples per pixel
    const s0 = Math.max(0, Math.floor(nowS - edge * spp)), s1 = Math.min(sig.length - 1, Math.floor(nowS));
    ctx.lineJoin = 'round'; ctx.lineWidth = 1.7; ctx.strokeStyle = '#1c1a16'; ctx.beginPath();
    for (let s = s0; s <= s1; s++) {
      const x = edge - (nowS - s) / spp, y = mid - (sig[s] / cur.scale) * PX_PER_MV;
      if (s === s0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    ctx.fillStyle = '#1c1a16'; ctx.beginPath(); ctx.arc(edge, mid - (sig[s1] / cur.scale) * PX_PER_MV, 3.5, 0, 7); ctx.fill();
    cur.hit = [];
    for (const b of cur.beats) {
      if (b.k < cur.warm) continue;
      if (b.p + POST > nowS) break;
      const x = edge - (nowS - b.p) / spp;
      if (x < -20) continue;
      classify(cur, b);
      const lab = CL[b.pred], yTop = mid - (sig[b.p] / cur.scale) * PX_PER_MV - 16;
      const wrong = showTruth && b.y !== b.pred;
      chip(x, Math.max(30, Math.min(yTop, mid - 40)), lab, COL[lab], wrong ? '#1c1a16' : (selected === b ? '#1c1a16' : null));
      if (showTruth) {
        ctx.font = '500 11px "Plex Mono", monospace'; ctx.textAlign = 'center';
        ctx.fillStyle = b.y === b.pred ? '#6d675a' : '#c4301e';
        ctx.fillText(b.y === b.pred ? 'ok' : 'label ' + CL[b.y], x, H - 26);
      }
      cur.hit.push({ x, b });
    }
  }
  function tally() {
    if (!cur) return;
    const seen = cur.beats.filter((b) => b.k >= cur.warm && b.k < cur.beats.length - 2 && b.p + POST <= t * FS);
    const c = { N: 0, S: 0, V: 0, F: 0 }; let ok = 0;
    seen.forEach((b) => { classify(cur, b); c[CL[b.pred]]++; if (b.pred === b.y) ok++; });
    $('tally').innerHTML = `<b>${seen.length}</b> beats · N ${c.N} · S ${c.S} · V ${c.V} · F ${c.F}` + (seen.length ? ` · agrees with label on <b>${pct(ok / seen.length, 1)}</b>` : '');
  }
  let tallyAt = 0;
  function frame(ts) {
    const dt = last ? Math.min(0.1, (ts - last) / 1000) : 0; last = ts;
    if (playing && cur) {
      t += dt * speed;
      if (t * FS >= cur.endS) { t = cur.startT; }
    }
    draw();
    if (ts - tallyAt > 250) { tally(); tallyAt = ts; }
    requestAnimationFrame(frame);
  }

  // ---------- inspector ----------
  function inspect(b) {
    selected = b; classify(cur, b);
    $('inspect').hidden = false; $('hint').hidden = true;
    const c = $('beatcv'), g = c.getContext('2d'), w = windowOf(cur, b);
    g.clearRect(0, 0, c.width, c.height);
    let mn = Infinity, mx = -Infinity; w.forEach((v) => { if (v < mn) mn = v; if (v > mx) mx = v; });
    const pad = 22, sy = (c.height - 2 * pad) / Math.max(0.5, mx - mn);
    g.strokeStyle = '#1c1a16'; g.lineWidth = 2; g.lineJoin = 'round'; g.beginPath();
    for (let i = 0; i < w.length; i++) { const x = (i / (w.length - 1)) * c.width, y = c.height - pad - (w[i] - mn) * sy; i ? g.lineTo(x, y) : g.moveTo(x, y); }
    g.stroke();
    g.strokeStyle = '#c4301e88'; g.setLineDash([4, 4]); g.beginPath(); const rx = (PRE / (w.length - 1)) * c.width; g.moveTo(rx, 0); g.lineTo(rx, c.height); g.stroke(); g.setLineDash([]);
    const sim = BeatModel.features(cur, b.k)[10];
    $('beatmeta').textContent = `${(b.p / FS).toFixed(1)} s into the excerpt · ${(PRE + POST) / FS * 1000 | 0} ms window · gap before ${(cur.rr(b.k, -1) / FS * 1000) | 0} ms, after ${(cur.rr(b.k, 1) / FS * 1000) | 0} ms · similarity to this patient's usual beat ${sim.toFixed(2)}`;
    $('bars').innerHTML = CL.map((k, i) => `<div class="bar ${i === b.y ? 'truth' : ''}" style="color:${COL[k]}"><span class="l">${k}</span><span class="t"><i style="width:${(b.probs[i] * 100).toFixed(1)}%"></i></span><span style="color:#1c1a16">${pct(b.probs[i], 1)}</span></div>`).join('');
    const ok = b.pred === b.y;
    $('verdict').textContent = ok ? `Called ${NAMES[CL[b.pred]]}. The cardiologist agrees.` : `Called ${NAMES[CL[b.pred]]}. The cardiologist labelled it ${NAMES[CL[b.y]]}. A miss.`;
    $('verdict').style.color = ok ? '#17695f' : '#c4301e';
  }
  cv.addEventListener('pointerdown', (e) => {
    if (!cur || !cur.hit) return;
    const r = cv.getBoundingClientRect(), x = e.clientX - r.left;
    let best = null, bd = 28;
    cur.hit.forEach((h) => { const d = Math.abs(h.x - x); if (d < bd) { bd = d; best = h.b; } });
    if (best) inspect(best);
  });

  // ---------- controls ----------
  function pick(i) {
    cur = RECS[i]; cur.beats.forEach((b, k) => { b.k = k; }); if (!cur._Z) { BeatModel.prepare(cur); cur.warm = cur.warmBeats; cur.startT = Math.max(2, cur.beats[cur.warm].p / FS - 2.5); cur.endS = cur.beats[cur.beats.length - 3].p + POST; } t = cur.startT; selected = null; $('inspect').hidden = true; $('hint').hidden = false;
    document.querySelectorAll('#records button').forEach((el, j) => el.setAttribute('aria-selected', j === i));
    if (reduceMotion) { playing = false; t = Math.min(cur.endS / FS - 1, cur.startT + 14); setPlayLabel(); }
  }
  function setPlayLabel() { $('play').textContent = playing ? 'Pause' : 'Play'; $('play').setAttribute('aria-pressed', playing); }
  $('play').onclick = () => { playing = !playing; setPlayLabel(); };
  $('truth').onchange = (e) => { showTruth = e.target.checked; };
  $('speed').onclick = (e) => { const s = e.target.dataset && e.target.dataset.s; if (!s) return; speed = +s; document.querySelectorAll('#speed button').forEach((b) => b.classList.toggle('on', b.dataset.s === s)); };

  // ---------- scorecard ----------
  function scorecard(m) {
    const f = $('figures');
    f.querySelector('[data-k=n]').textContent = m.n.toLocaleString('en-US');
    f.querySelector('[data-k=acc]').textContent = pct(m.acc, 1);
    f.querySelector('[data-k=vsens]').textContent = pct(m.classes[2].sens, 1);
    f.querySelector('[data-k=vppv]').textContent = pct(m.classes[2].ppv, 1);
    let h = '<span class="h"></span>' + CL.map((k) => `<span class="h">${k}</span>`).join('');
    m.cm.forEach((row, i) => {
      const tot = row.reduce((a, v) => a + v, 0) || 1;
      h += `<span class="h">${CL[i]}</span>` + row.map((v, j) => {
        const a = v / tot, base = i === j ? '23,105,95' : '196,48,30';
        return `<span class="c" style="background:rgba(${base},${(0.07 + a * 0.85).toFixed(2)});color:${a > 0.5 ? '#fff' : '#1c1a16'}">${v.toLocaleString('en-US')}<span>${pct(a, a < 0.01 && v ? 2 : 1)}</span></span>`;
      }).join('');
    });
    $('cm').innerHTML = h;
    $('perclass').innerHTML = '<table><thead><tr><th>Class</th><th>Found</th><th>Precision</th><th>F1</th><th class="dim">F1 baseline</th></tr></thead><tbody>' +
      m.classes.map((c) => `<tr><td class="cl" style="color:${COL[c.c]}">${c.c} <span class="dim" style="font-weight:400">${c.n.toLocaleString('en-US')}</span></td><td>${pct(c.sens)}</td><td>${pct(c.ppv)}</td><td>${c.f1.toFixed(3)}</td><td class="dim">${c.base_f1.toFixed(3)}</td></tr>`).join('') + '</tbody></table>';
    $('compare').textContent = `"Found" is sensitivity: the share of true beats of that class the model caught. The baseline is gradient-boosted trees on the same timing features and a coarse beat shape, trained on the same patients. Overall accuracy: network ${pct(m.acc)}, baseline ${pct(m.base_acc)}.`;
    $('callout').innerHTML = m.callout;
  }

  async function init() {
    try {
      const [mo, me, re] = await Promise.all(['weights', 'metrics', 'records'].map((n) => fetch(`data/${n}.json`).then((r) => r.json())));
      MODEL = mo; METRICS = me; RECS = re; RECS.forEach((r) => { r.rr = (k, d) => (d < 0 ? (k > 0 ? r.beats[k].p - r.beats[k - 1].p : 0) : (k < r.beats.length - 1 ? r.beats[k + 1].p - r.beats[k].p : 0)); });
    } catch (e) {
      $('hint').textContent = 'Could not load the model data. Serve this folder over http and reload.'; return;
    }
    scorecard(METRICS);
    const rb = $('records');
    RECS.forEach((r, i) => { const b = document.createElement('button'); b.setAttribute('role', 'tab'); b.innerHTML = `<b>Record ${r.id}</b>${r.title}`; b.onclick = () => pick(i); rb.appendChild(b); });
    resize(); addEventListener('resize', resize); pick(0); requestAnimationFrame(frame);
  }
  init();
})();

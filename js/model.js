// Beat classifier, in plain JavaScript: patient-adaptive features computed
// from the recording itself, then a small neural network (5 models averaged).
// Mirrors train/feats2.py and train/mlp.py; test/parity.test.js checks it
// against the Python implementation on real beats.
(function (root) {
  'use strict';
  const CLASSES = ['N', 'S', 'V', 'F'];
  const PRE = 96, POST = 144, LEN = PRE + POST, K = 30;
  const nanmean = (a) => { let s = 0, n = 0; for (const v of a) if (!Number.isNaN(v)) { s += v; n++; } return n ? s / n : NaN; };
  const median = (a) => {
    const b = Float64Array.from(a).sort(); const n = b.length;
    return n % 2 ? b[(n - 1) / 2] : (b[n / 2 - 1] + b[n / 2]) / 2;
  };
  const nanmedian = (a) => median(a.filter((v) => !Number.isNaN(v)));
  const clip = (v, lo, hi) => Math.min(hi, Math.max(lo, v));

  // Per-recording context: standardised beat windows and RR series.
  function prepare(rec) {
    const n = rec.beats.length, sig = rec.sig, sc = rec.scale;
    rec._raw = new Array(n); rec._Z = new Array(n);
    for (let k = 0; k < n; k++) {
      const p = rec.beats[k].p, w = new Float64Array(LEN);
      for (let i = 0; i < LEN; i++) w[i] = sig[p - PRE + i] / sc;
      rec._raw[k] = w;
      let mu = 0; for (let i = 0; i < LEN; i++) mu += w[i]; mu /= LEN;
      let v = 0; for (let i = 0; i < LEN; i++) v += (w[i] - mu) * (w[i] - mu);
      const sd = Math.sqrt(v / LEN) + 1e-6, z = new Float64Array(LEN);
      for (let i = 0; i < LEN; i++) z[i] = (w[i] - mu) / sd;
      rec._Z[k] = z;
    }
    rec._rr = []; for (let k = 0; k + 1 < n; k++) rec._rr.push(rec.beats[k + 1].p - rec.beats[k].p);
    rec._fill = nanmedian(rec._rr);
  }

  const widthFrac = (z) => { let m = 0; for (const v of z) m = Math.max(m, Math.abs(v)); let c = 0; for (const v of z) if (Math.abs(v) > 0.5 * m) c++; return c / LEN; };
  const argmaxAbs = (z) => { let b = 0, m = -1; for (let i = 0; i < z.length; i++) if (Math.abs(z[i]) > m) { m = Math.abs(z[i]); b = i; } return b / LEN; };

  function features(rec, k) {
    if (!rec._Z) prepare(rec);
    const n = rec.beats.length, rr = rec._rr, fill = rec._fill, nn = (v) => (Number.isNaN(v) ? fill : v);
    const pre = nn(k > 0 ? rr[k - 1] : NaN), post = nn(k < n - 1 ? rr[k] : NaN);
    const prepre = nn(k > 1 ? rr[k - 2] : NaN), postpost = nn(k < n - 2 ? rr[k + 1] : NaN);
    const loc = nn(k > 0 ? nanmean(rr.slice(Math.max(0, k - 10), k)) : NaN);
    const med = nn(k > 0 ? nanmedian(rr.slice(Math.max(0, k - 20), k)) : NaN);
    const f = [pre / loc, post / loc, pre / post, prepre / loc, postpost / loc, (pre + post) / (2 * loc), pre / med, post / med, Math.abs(pre - post) / loc].map((v) => clip(v, 0.1, 6));
    const w = rec._raw[k]; let mn = Infinity, mx = -Infinity; for (const v of w) { if (v < mn) mn = v; if (v > mx) mx = v; }
    f.push(Math.log((mx - mn) / rec.medPtp + 1e-3));
    // patient-adaptive: compare with the median of this patient's previous K beats
    const z = rec._Z[k]; let tpl;
    if (k > 0) {
      const lo = Math.max(0, k - K), cnt = k - lo; tpl = new Float64Array(LEN);
      const col = new Float64Array(cnt);
      for (let i = 0; i < LEN; i++) { for (let j = 0; j < cnt; j++) col[j] = rec._Z[lo + j][i]; tpl[i] = median(col); }
    } else tpl = z;
    let corr = 1;
    if (k > 0) {
      let mz = 0, mt = 0; for (let i = 0; i < LEN; i++) { mz += z[i]; mt += tpl[i]; } mz /= LEN; mt /= LEN;
      let sxy = 0, sxx = 0, syy = 0; for (let i = 0; i < LEN; i++) { const a = z[i] - mz, b = tpl[i] - mt; sxy += a * b; sxx += a * a; syy += b * b; }
      corr = sxy / Math.sqrt(sxx * syy);
    }
    let sq = 0, mxd = 0; const diff = new Float64Array(LEN);
    for (let i = 0; i < LEN; i++) { diff[i] = z[i] - tpl[i]; sq += diff[i] * diff[i]; mxd = Math.max(mxd, Math.abs(diff[i])); }
    f.push(corr, Math.sqrt(sq / LEN), mxd, widthFrac(z) - widthFrac(tpl), argmaxAbs(z) - argmaxAbs(tpl));
    return f;
  }

  function mlp(layers, x) {
    let h = x;
    layers.forEach((L, li) => {
      const out = new Float64Array(L.o);
      for (let o = 0; o < L.o; o++) { let a = L.b[o]; const base = o * L.i; for (let i = 0; i < L.i; i++) a += h[i] * L.w[base + i]; out[o] = li < layers.length - 1 ? Math.max(0, a) : a; }
      h = out;
    });
    return h;
  }
  function softmax(z) { const m = Math.max.apply(null, z); const e = Array.from(z, (v) => Math.exp(v - m)); const s = e.reduce((a, v) => a + v, 0); return e.map((v) => v / s); }
  function predictFeatures(M, f) {
    const x = f.map((v, i) => (v - M.mu[i]) / M.sd[i]), acc = [0, 0, 0, 0];
    for (const net of M.nets) { const p = softmax(mlp(net, x)); for (let c = 0; c < 4; c++) acc[c] += p[c] / M.nets.length; }
    return acc;
  }
  const predict = (M, rec, k) => predictFeatures(M, features(rec, k));

  const api = { CLASSES, PRE, POST, LEN, prepare, features, predictFeatures, predict };
  if (typeof module !== 'undefined' && module.exports) module.exports = api; else root.BeatModel = api;
})(typeof self !== 'undefined' ? self : this);

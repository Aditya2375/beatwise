import numpy as np, json
import prep
from feats2 import rr_feats
PICK = [(200, 'Frequent ventricular beats, mixed in with normal ones'), (233, 'Long runs of ventricular beats'), (213, 'Fusion beats, the rarest class'),
        (100, 'Clean rhythm with the odd early beat'), (222, 'Supraventricular beats on an irregular rhythm'), (232, 'The one it gets wrong: sick-sinus rhythm')]
WARM, SPAN = 40, 40 * 360
d = np.load('data/ds2.npz'); out = []; fx = []
for rid, title in PICK:
    x, idx, sym = prep.load(str(rid))
    m = d['rec'] == rid; pos = d['pos'][m]; y = d['y'][m]; X = d['X'][m]
    ptp = X.max(1) - X.min(1); med = float(np.median(ptp))
    # window rule: the 40 s stretch with the most beats that are not normal
    nn = np.where(y != 0)[0]
    best, bi = -1, WARM
    for i in range(WARM, len(pos) - 3):
        t0 = pos[i]; c = ((pos[i:] < t0 + SPAN) & (y[i:] != 0)).sum()
        if c > best: best, bi = c, i
    i0, i1 = bi - WARM, int(np.searchsorted(pos, pos[bi] + SPAN)) + 2
    i1 = min(i1, len(pos))
    s0 = max(0, pos[i0] - 120); s1 = min(len(x), pos[i1 - 1] + 200)
    sig = np.round(x[s0:s1] * 1000).astype(int)
    beats = [dict(p=int(pos[i] - s0), y=int(y[i])) for i in range(i0, i1)]
    out.append(dict(id=rid, title=title, scale=1000, medPtp=med, warmBeats=WARM, sig=sig.tolist(), beats=beats))
    # python reference features on this exact subset
    sub_pos = np.array([b['p'] for b in beats]); rr = rr_feats(sub_pos, np.zeros(len(beats), int))
    fx.append(dict(id=rid, rr=rr[WARM:WARM + 25].tolist()))
json.dump(out, open('web/data/records.json', 'w'), separators=(',', ':'))
json.dump(fx, open('train/parity_rr.json', 'w'))
print([(r['id'], len(r['beats']), len(r['sig'])) for r in out])

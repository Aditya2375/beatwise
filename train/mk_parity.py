import numpy as np, json
from feats2 import rr_feats, tpl_arrays
recs = json.load(open('web/data/records.json')); W = json.load(open('web/data/weights.json'))
out = []
for r in recs:
    sig = np.array(r['sig']) / r['scale']; pos = np.array([b['p'] for b in r['beats']])
    X = np.stack([sig[p - 96:p + 144] for p in pos]).astype(np.float64)
    rr = rr_feats(pos, np.zeros(len(pos), int)); ptp = X.max(1) - X.min(1); amp = np.log(ptp / r['medPtp'] + 1e-3)[:, None]
    F, _, _ = tpl_arrays(X.astype(np.float32), np.zeros(len(pos), int))
    f = np.concatenate([rr, amp, F], 1)
    mu = np.array(W['mu']); sd = np.array(W['sd']); x = (f - mu) / sd; P = 0
    for net in W['nets']:
        h = x
        for li, L in enumerate(net):
            h = h @ np.array(L['w']).reshape(L['o'], L['i']).T + np.array(L['b']); h = np.maximum(h, 0) if li < 2 else h
        e = np.exp(h - h.max(1, keepdims=True)); P = P + e / e.sum(1, keepdims=True) / len(W['nets'])
    ks = list(range(r['warmBeats'], len(pos) - 2))
    out.append(dict(id=r['id'], ks=ks, f=f[ks].tolist(), p=P[ks].tolist()))
json.dump(out, open('test/parity.json', 'w'))
# per-excerpt python accuracy vs labels for the page
for o, r in zip(out, recs):
    y = np.array([r['beats'][k]['y'] for k in o['ks']]); pr = np.array(o['p']).argmax(1); print(r['id'], len(y), round(float((y == pr).mean()), 3))

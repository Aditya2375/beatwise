import numpy as np
FS = 360
def rr_feats(pos, rec):
    out = np.zeros((len(pos), 9), np.float32)
    for r in np.unique(rec):
        m = np.where(rec == r)[0]; p = pos[m].astype(np.float64); n = len(p)
        rr = np.diff(p)                      # rr[i] = p[i+1]-p[i]
        pre = np.r_[np.nan, rr]; post = np.r_[rr, np.nan]
        prepre = np.r_[np.nan, np.nan, rr[:-1]] if n > 1 else pre; postpost = np.r_[rr[1:], np.nan, np.nan] if n > 1 else post
        prepre = prepre[:n]; postpost = postpost[:n]
        loc = np.array([np.nanmean(rr[max(0, i - 10):i]) if i > 0 else np.nan for i in range(n)])
        med = np.array([np.nanmedian(rr[max(0, i - 20):i]) if i > 0 else np.nan for i in range(n)])
        fill = np.nanmedian(rr)
        for a in (pre, post, prepre, postpost, loc, med): a[np.isnan(a)] = fill
        f = np.stack([pre / loc, post / loc, pre / post, prepre / loc, postpost / loc, (pre + post) / (2 * loc), pre / med, post / med, np.abs(pre - post) / loc], 1)
        out[m] = np.clip(f, 0.1, 6)
    return out
def morph(X, rec):
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-6
    Z = (X - mu) / sd
    dec = Z.reshape(len(Z), 60, 4).mean(2)
    ptp = (X.max(1) - X.min(1)).astype(np.float64); amp = np.zeros(len(X), np.float32)
    for r in np.unique(rec):
        m = rec == r; amp[m] = np.log(ptp[m] / (np.median(ptp[m]) + 1e-6) + 1e-3)
    return dec.astype(np.float32), amp[:, None]
def build(path):
    d = np.load(path); R = rr_feats(d['pos'], d['rec']); M, A = morph(d['X'], d['rec'])
    return np.concatenate([R, A], 1), M, d['y'], d['rec']

def template_feats(path, K=30):
    """Patient-adaptive: compare each beat with the median of this patient's previous K beats (online-computable)."""
    d = np.load(path); return tpl_arrays(d['X'], d['rec'], K)

def tpl_arrays(X, rec, K=30):
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-6; Z = ((X - mu) / sd).astype(np.float32)
    F = np.zeros((len(Z), 5), np.float32); D = np.zeros((len(Z), 60), np.float32); T = np.zeros_like(Z)
    for r in np.unique(rec):
        m = np.where(rec == r)[0]
        for k, i in enumerate(m):
            lo = max(0, k - K)
            tpl = np.median(Z[m[lo:k]], 0) if k > 0 else Z[i]
            T[i] = tpl; z = Z[i]
            corr = float(np.corrcoef(z, tpl)[0, 1]) if k > 0 else 1.0
            diff = z - tpl
            F[i] = [corr, np.sqrt((diff ** 2).mean()), np.abs(diff).max(), (np.abs(z) > 0.5 * np.abs(z).max()).sum() / 240.0 - (np.abs(tpl) > 0.5 * np.abs(tpl).max()).sum() / 240.0, np.abs(z).argmax() / 240.0 - np.abs(tpl).argmax() / 240.0]
            D[i] = diff.reshape(60, 4).mean(1)
    return F, D, T

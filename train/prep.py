"""Build beat windows from MIT-BIH with the inter-patient split (de Chazal et al. 2004)."""
import numpy as np, wfdb, scipy.signal as sg, json, sys
DS1 = [101,106,108,109,112,114,115,116,118,119,122,124,201,203,205,207,208,209,215,220,223,230]
DS2 = [100,103,105,111,113,117,121,123,200,202,210,212,213,214,219,221,222,228,231,232,233,234]
AAMI = {}
for s in 'NLRej': AAMI[s]=0
for s in 'AaJS': AAMI[s]=1
for s in 'VE': AAMI[s]=2
AAMI['F']=3
PRE, POST = 96, 144   # samples at 360 Hz around the R peak
FS = 360

def baseline(x):
    # median filters, 200 ms then 600 ms, subtract (standard baseline wander removal)
    k1 = int(0.2*FS)//2*2+1; k2 = int(0.6*FS)//2*2+1
    b = sg.medfilt(sg.medfilt(x, k1), k2)
    return x - b

def load(rec):
    r = wfdb.rdrecord(f'data/mitdb/{rec}')
    a = wfdb.rdann(f'data/mitdb/{rec}', 'atr')
    ch = r.sig_name.index('MLII')
    x = r.p_signal[:, ch].astype(np.float64)
    x = baseline(x)
    # light low-pass at 45 Hz (mains/EMG)
    b, a2 = sg.butter(3, 45/(FS/2)); x = sg.filtfilt(b, a2, x)
    return x, np.array(a.sample), np.array(a.symbol)

def beats(rec):
    x, idx, sym = load(rec)
    keep = [i for i,s in enumerate(sym) if s in AAMI]
    X, y, rr, pos = [], [], [], []
    allr = idx[keep]
    for j, i in enumerate(keep):
        p = idx[i]
        if p-PRE < 0 or p+POST > len(x): continue
        w = x[p-PRE:p+POST]
        pre = (allr[j]-allr[j-1]) if j>0 else np.nan
        post = (allr[j+1]-allr[j]) if j+1<len(allr) else np.nan
        loc = np.nanmean(allr[max(0,j-10):j+1][1:] - allr[max(0,j-10):j+1][:-1]) if j>1 else np.nan
        X.append(w); y.append(AAMI[sym[i]]); rr.append([pre, post, loc]); pos.append(p)
    return np.array(X, np.float32), np.array(y), np.array(rr, np.float64), np.array(pos)

def build(recs):
    Xs, ys, rrs, ids, ps = [], [], [], [], []
    for r in recs:
        X, y, rr, p = beats(str(r))
        # fill edges, then normalise RR features: pre/post relative to the local mean, plus the mean in seconds
        m = np.nanmedian(rr[:,2]); rr = np.where(np.isnan(rr), m, rr)
        f = np.stack([rr[:,0]/rr[:,2], rr[:,1]/rr[:,2], rr[:,2]/FS], 1).astype(np.float32)
        Xs.append(X); ys.append(y); rrs.append(f); ids.append(np.full(len(y), r)); ps.append(p)
    return np.concatenate(Xs), np.concatenate(ys), np.concatenate(rrs), np.concatenate(ids), np.concatenate(ps)

if __name__ == '__main__':
    for name, recs in (('ds1', DS1), ('ds2', DS2)):
        X, y, f, ids, ps = build(recs)
        np.savez_compressed(f'data/{name}.npz', X=X, y=y, f=f, rec=ids, pos=ps)
        print(name, X.shape, np.bincount(y, minlength=4))

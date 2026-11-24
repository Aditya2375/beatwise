"""Final model: 5-seed MLP ensemble on patient-adaptive features, trained on all of DS1, scored once on DS2."""
import numpy as np, json, torch
from sklearn.metrics import confusion_matrix
from feats2 import build, template_feats
from mlp import fit, predict
CL = ['N', 'S', 'V', 'F']
def feats(path):
    T, M, y, rec = build(path); F, D, _ = template_feats(path)
    return np.concatenate([T, F], 1).astype(np.float32), y, rec
X1, y1, r1 = feats('data/ds1.npz'); X2, y2, r2 = feats('data/ds2.npz')
np.savez('data/feat_ds2.npz', X=X2, y=y2, rec=r2)
mu = X1.mean(0); sd = X1.std(0) + 1e-6
nets = [fit(X1, y1, ep=40, seed=s, mu=mu, sd=sd)[0] for s in range(5)]
P = np.mean([predict(n, X2, mu, sd) for n in nets], 0); pred = P.argmax(1)
np.save('train/pred_ds2_final.npy', pred); np.save('train/prob_ds2_final.npy', P)
base = np.load('train/baseline_pred_ds2.npy')
def stats(pr):
    cm = confusion_matrix(y2, pr, labels=range(4)); out = []
    for i in range(4):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        out.append(dict(n=int(cm[i].sum()), sens=float(tp / max(1, tp + fn)), ppv=float(tp / max(1, tp + fp)), f1=float(2 * tp / max(1, 2 * tp + fp + fn))))
    return cm, out, float((pr == y2).mean())
cm, cls, acc = stats(pred); bcm, bcls, bacc = stats(base)
for i, c in enumerate(cls): c.update(c=CL[i], base_f1=bcls[i]['f1'], base_sens=bcls[i]['sens'], base_ppv=bcls[i]['ppv'])
per = {int(r): dict(n=int((r2 == r).sum()), acc=float((pred[r2 == r] == y2[r2 == r]).mean()), counts=np.bincount(y2[r2 == r], minlength=4).tolist()) for r in np.unique(r2)}
json.dump(dict(n=int(len(y2)), acc=acc, base_acc=bacc, classes=cls, cm=cm.tolist(), base_cm=bcm.tolist(), per_record=per, callout=''), open('train/metrics_raw.json', 'w'), indent=1)
print(cm); print('acc', acc, 'base', bacc); [print(c) for c in cls]
def lin(l): return dict(w=np.round(l.weight.detach().numpy().ravel(), 5).tolist(), b=np.round(l.bias.detach().numpy(), 5).tolist(), i=int(l.in_features), o=int(l.out_features))
M = dict(mu=mu.tolist(), sd=sd.tolist(), nets=[[lin(n[0]), lin(n[3]), lin(n[6])] for n in nets])
json.dump(M, open('web/data/weights.json', 'w'), separators=(',', ':'))

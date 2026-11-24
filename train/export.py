"""Evaluate on DS2 once, export folded weights, metrics and browser excerpts."""
import numpy as np, torch, json, sys
from common import *
from sklearn.metrics import confusion_matrix
import prep
seed = sys.argv[1] if len(sys.argv) > 1 else '0'
net = BeatNet(); net.load_state_dict(torch.load(f'train/net_s{seed}_last.pt')); net.eval()
b = np.load('data/ds2.npz'); Z, ff = prep_inputs(b['X'], b['f'], b['rec']); y = b['y']
with torch.no_grad():
    P = np.concatenate([torch.softmax(net(torch.tensor(Z[i:i+1024]), torch.tensor(ff[i:i+1024])), 1).numpy() for i in range(0, len(Z), 1024)])
pred = P.argmax(1); np.save('train/pred_ds2.npy', pred)
base = np.load('train/baseline_pred_ds2.npy')
def stats(pr):
    cm = confusion_matrix(y, pr, labels=range(4)); out = []
    for i in range(4):
        tp = cm[i, i]; fn = cm[i].sum() - tp; fp = cm[:, i].sum() - tp
        out.append(dict(n=int(cm[i].sum()), sens=float(tp / max(1, tp + fn)), ppv=float(tp / max(1, tp + fp)), f1=float(2 * tp / max(1, 2 * tp + fp + fn))))
    return cm, out, float((pr == y).mean())
cm, cls, acc = stats(pred); bcm, bcls, bacc = stats(base)
for i, c in enumerate(cls): c['c'] = CLASSES[i]; c['base_f1'] = bcls[i]['f1']; c['base_sens'] = bcls[i]['sens']; c['base_ppv'] = bcls[i]['ppv']
per = {}
for r in np.unique(b['rec']):
    m = b['rec'] == r; per[int(r)] = dict(n=int(m.sum()), acc=float((pred[m] == y[m]).mean()), counts=np.bincount(y[m], minlength=4).tolist())
metrics = dict(n=int(len(y)), acc=acc, base_acc=bacc, classes=cls, cm=cm.tolist(), base_cm=bcm.tolist(), per_record=per, callout='')
json.dump(metrics, open('train/metrics_raw.json', 'w'), indent=1)
print(cm); print(json.dumps(cls, indent=0)); print('acc', acc, 'base', bacc)
def lay(conv, gn):
    return dict(w=np.round(conv.weight.detach().numpy().ravel(), 5).tolist(), b=[0.0] * conv.out_channels, cout=int(conv.out_channels),
                g=np.round(gn.weight.detach().numpy(), 5).tolist(), be=np.round(gn.bias.detach().numpy(), 5).tolist())
M = dict(stem=lay(net.stem[0], net.stem[1]), blocks=[])
for blk in net.blocks:
    d = dict(stride=int(blk.c1.stride[0]), c1=lay(blk.c1, blk.b1), c2=lay(blk.c2, blk.b2))
    if blk.sk is not None: d['sk'] = lay(blk.sk[0], blk.sk[1])
    M['blocks'].append(d)
M['fc1'] = dict(w=np.round(net.fc1.weight.detach().numpy().ravel(), 5).tolist(), b=np.round(net.fc1.bias.detach().numpy(), 5).tolist(), out=64)
M['fc2'] = dict(w=np.round(net.fc2.weight.detach().numpy().ravel(), 5).tolist(), b=np.round(net.fc2.bias.detach().numpy(), 5).tolist())
json.dump(M, open('web/data/weights.json', 'w'), separators=(',', ':'))
# parity fixtures: 40 DS2 beats
idx = np.random.RandomState(1).choice(len(y), 40, replace=False)
fx = [dict(window=b['X'][i].tolist(), rr=b['f'][i].tolist(), rec=int(b['rec'][i]), probs=P[i].tolist()) for i in idx]
json.dump(dict(items=fx, note='raw windows, f = [pre/loc, post/loc, loc/360]; amp feature recomputed in test from ptp and per-record median'), open('train/parity.json', 'w'))

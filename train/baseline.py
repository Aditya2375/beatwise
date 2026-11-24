"""Classical baseline: gradient boosting on RR-interval features plus a coarse morphology descriptor."""
import numpy as np, json
from sklearn.ensemble import HistGradientBoostingClassifier
from common import *
def feats(d):
    Z, ff = prep_inputs(d['X'], d['f'], d['rec'])
    # morphology: beat decimated to 40 samples, plus width/area descriptors
    dec = Z.reshape(len(Z), 40, 6).mean(2)
    area = np.abs(Z).mean(1, keepdims=True); pk = np.abs(Z).argmax(1)[:, None] / 240.0
    return np.concatenate([ff, dec, area, pk], 1)
a = np.load('data/ds1.npz'); b = np.load('data/ds2.npz')
tr = np.ones(len(a['y']), bool)  # same training patients as the final network (all of DS1)
Xa, Xb = feats(a), feats(b)
cnt = np.bincount(a['y'][tr], minlength=4); sw = ((cnt.sum() / cnt) ** 0.5)[a['y'][tr]]
clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15, random_state=0)
clf.fit(Xa[tr], a['y'][tr], sample_weight=sw)
p = clf.predict(Xb); np.save('train/baseline_pred_ds2.npy', p)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(b['y'], p))

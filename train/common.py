import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
CLASSES = ['N','S','V','F']
VAL_RECS = [101,118,208,215,223]

def prep_inputs(X, f, rec):
    mu = X.mean(1, keepdims=True); sd = X.std(1, keepdims=True) + 1e-6
    Z = ((X - mu) / sd).astype(np.float32)
    ptp = (X.max(1) - X.min(1)).astype(np.float64)
    amp = np.zeros(len(X), np.float32)
    for r in np.unique(rec):
        m = rec == r
        amp[m] = np.log(ptp[m] / (np.median(ptp[m]) + 1e-6) + 1e-3)
    # no absolute heart-rate feature: it identifies the patient and the model latched onto it
    ff = np.concatenate([np.log(np.clip(f[:, :2], 0.2, 5.0)), amp[:, None]], 1).astype(np.float32)
    return Z, ff

class Block(nn.Module):
    def __init__(s, cin, cout, stride):
        super().__init__()
        s.c1 = nn.Conv1d(cin, cout, 7, stride, 3, bias=False); s.b1 = nn.GroupNorm(4, cout)
        s.c2 = nn.Conv1d(cout, cout, 7, 1, 3, bias=False); s.b2 = nn.GroupNorm(4, cout)
        s.sk = None
        if stride != 1 or cin != cout:
            s.sk = nn.Sequential(nn.Conv1d(cin, cout, 1, stride, bias=False), nn.GroupNorm(4, cout))
    def forward(s, x):
        y = F.relu(s.b1(s.c1(x))); y = s.b2(s.c2(y))
        return F.relu(y + (x if s.sk is None else s.sk(x)))

class BeatNet(nn.Module):
    def __init__(s, nf=3, w=(16, 32, 48, 64)):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv1d(1, w[0], 9, 1, 4, bias=False), nn.GroupNorm(4, w[0]), nn.ReLU())
        s.blocks = nn.Sequential(Block(w[0], w[1], 2), Block(w[1], w[2], 2), Block(w[2], w[3], 2))
        s.fc1 = nn.Linear(w[3] * 2 + nf, 64); s.drop = nn.Dropout(0.4); s.fc2 = nn.Linear(64, 4)
    def forward(s, x, f):
        h = s.blocks(s.stem(x.unsqueeze(1)))
        h = torch.cat([h.mean(2), h.amax(2), f], 1)
        return s.fc2(s.drop(F.relu(s.fc1(h))))

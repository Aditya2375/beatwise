import numpy as np, torch, torch.nn as nn, torch.nn.functional as F_
def make(nin): return nn.Sequential(nn.Linear(nin, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.3), nn.Linear(64, 4))
def fit(X, y, ep=40, seed=0, mu=None, sd=None):
    torch.manual_seed(seed); np.random.seed(seed)
    mu = X.mean(0) if mu is None else mu; sd = X.std(0) + 1e-6 if sd is None else sd
    Xt = torch.tensor((X - mu) / sd, dtype=torch.float32); yt = torch.tensor(y)
    net = make(X.shape[1]); opt = torch.optim.AdamW(net.parameters(), 2e-3, weight_decay=1e-2)
    cnt = np.bincount(y, minlength=4); w = torch.tensor((cnt.sum() / cnt) ** 0.5, dtype=torch.float32); w /= w.mean()
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, ep * (len(y) // 256 + 1))
    for e in range(ep):
        net.train(); perm = torch.randperm(len(y))
        for i in range(0, len(y), 256):
            b = perm[i:i + 256]; x = Xt[b] + 0.1 * torch.randn(len(b), Xt.shape[1])
            loss = F_.cross_entropy(net(x), yt[b], weight=w, label_smoothing=0.02); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    net.eval(); return net, mu, sd
def predict(net, X, mu, sd):
    with torch.no_grad(): return torch.softmax(net(torch.tensor((X - mu) / sd, dtype=torch.float32)), 1).numpy()

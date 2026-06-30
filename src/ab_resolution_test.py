"""
Controlled A/B #3: input resolution. 128x64 vs 128x128 vs 224x224.

All historical ~69-70% runs used 224x224; the small 128x64 models cap ~64%.
This isolates whether resolution is the lever. Same compact SE-residual model
(AdaptiveAvgPool head works at any size), same split, seed, light recipe — only
the G-channel decode size differs. One resolution held in RAM at a time.
"""
import gc, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA = Path('../data/raw')
SEED = 0
EPOCHS = 20
BATCH = 128
RESOLUTIONS = [(128, 64), (128, 128), (224, 224)]

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('device:', device)

_DF = pd.read_csv(DATA / 'train.csv')
_Y = torch.tensor(_DF['label'].values - 1)


def decode_all(H, W):
    X = torch.empty(len(_DF), 1, H, W)
    for i, row in enumerate(_DF.itertuples(index=False)):
        g = np.array(Image.open(DATA / 'train' / row.id))[:, :, 1].astype(np.float32) / 255.0
        t = torch.from_numpy(g)[None, None]
        X[i] = F.interpolate(t, size=(H, W), mode='bilinear', align_corners=False)[0]
        if (i + 1) % 5000 == 0:
            print(f'  decoded {i+1}/{len(_DF)} @ {H}x{W}')
    return X


class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__(); s = max(1, c // r); self.c = c
        self.fc = nn.Sequential(nn.Linear(c, s, bias=False), nn.SiLU(),
                                nn.Linear(s, c, bias=False), nn.Sigmoid())
    def forward(self, x):
        return x * self.fc(x.mean((2, 3))).view(x.size(0), self.c, 1, 1)

class CBA(nn.Module):
    def __init__(self, i, o, s=1):
        super().__init__()
        self.c = nn.Conv2d(i, o, 3, s, 1, bias=False); self.b = nn.BatchNorm2d(o); self.a = nn.SiLU(True)
    def forward(self, x): return self.a(self.b(self.c(x)))

class Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(c)
        self.se = SE(c); self.a = nn.SiLU(True); nn.init.zeros_(self.b2.weight)
    def forward(self, x):
        o = self.a(self.b1(self.c1(x))); o = self.se(self.b2(self.c2(o)))
        return self.a(x + o)

class Net(nn.Module):
    def __init__(self, ch=(32, 64, 128, 192), nc=5, p=0.3):
        super().__init__()
        self.stem = CBA(1, ch[0])
        blocks, prev = [], ch[0]
        for c in ch:
            blocks += [CBA(prev, c, 2), Res(c)]; prev = c
        self.body = nn.Sequential(*blocks)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(prev, prev), nn.SiLU(True), nn.Dropout(p), nn.Linear(prev, nc))
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, Res): nn.init.zeros_(m.b2.weight)
    def forward(self, x): return self.head(self.body(self.stem(x)))


class AugDS(Dataset):
    def __init__(self, X, y, mean, std, train):
        self.X, self.y, self.m, self.s, self.train = X, y, mean, std, train
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.train:
            c = 0.8 + 0.4 * torch.rand(1).item(); b = (torch.rand(1).item() - 0.5) * 0.1
            x = ((x - 0.5) * c + 0.5 + b).clamp_(0, 1)
            x.add_(torch.randn_like(x) * 0.02).clamp_(0, 1)
        return (x - self.m) / self.s, int(self.y[i])


def run_arm(name, X, y, tr, va):
    torch.manual_seed(SEED); np.random.seed(SEED)
    mean = X[tr].mean().item(); std = X[tr].std().item()
    gen = torch.Generator().manual_seed(SEED)
    tl = DataLoader(AugDS(X[tr], y[tr], mean, std, True), batch_size=BATCH,
                    shuffle=True, num_workers=0, drop_last=True, generator=gen)
    Xva = ((X[va] - mean) / std); yva = y[va]

    model = Net().to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=1.5e-3, total_steps=EPOCHS * len(tl), pct_start=0.25)

    best = 0.0
    for ep in range(EPOCHS):
        model.train(); tr_c = torch.zeros((), device=device); tr_n = 0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step(); sched.step()
            with torch.no_grad(): tr_c += (out.argmax(1) == yb).sum()
            tr_n += yb.size(0)
        model.eval(); correct = 0
        with torch.no_grad():
            for s in range(0, len(Xva), 256):
                p = model(Xva[s:s+256].to(device)).argmax(1).cpu()
                correct += (p == yva[s:s+256]).sum().item()
        acc = 100 * correct / len(Xva); best = max(best, acc)
        ta = 100 * tr_c.item() / tr_n
        print(f'  [{name}] ep {ep+1:2d}/{EPOCHS}  train {ta:5.2f}  val {acc:5.2f}  (best {best:5.2f})', flush=True)
    return best


def main():
    idx = np.arange(len(_Y))
    tr, va = train_test_split(idx, test_size=0.2, stratify=_Y.numpy(), random_state=42)
    tr, va = torch.from_numpy(tr), torch.from_numpy(va)

    results = {}
    t0 = time.time()
    for (H, W) in RESOLUTIONS:
        tag = f'{H}x{W}'
        print(f'\n=== RESOLUTION {tag} ===')
        td = time.time(); X = decode_all(H, W); print(f'  decoded in {time.time()-td:.0f}s')
        results[tag] = run_arm(tag, X, _Y, tr, va)
        del X; gc.collect()

    print('\n==================== A/B RESULT (resolution) ====================')
    base = results[f'{RESOLUTIONS[0][0]}x{RESOLUTIONS[0][1]}']
    for tag, acc in results.items():
        print(f'{tag:>9}  best val: {acc:5.2f}%   (Δ vs {RESOLUTIONS[0][0]}x{RESOLUTIONS[0][1]}: {acc-base:+.2f} pp)')
    print(f'(elapsed {time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()

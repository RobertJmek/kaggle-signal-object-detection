"""
Controlled A/B #4: model capacity. Small vs Medium(production ~7M) vs Large.

Same input (G channel 128x64), split, seed, and the Run-18 "let it fit" recipe
(light aug, no MixUp, EMA). Only channels/depths of SignalNetV2 differ. This
tells us whether the ~64% ceiling is capacity-bound or recipe/data-bound.

Note: 30-epoch proxy at fixed budget — larger nets converge slower, so treat the
ranking as "best per this budget", not each model's asymptotic ceiling.
"""
import gc, time, copy
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
IMG_H, IMG_W = 128, 64
SEED = 42
EPOCHS = 30
BATCH = 256
CACHE = Path('/tmp/ab_g_12864.pt')

CONFIGS = {
    'S  (32-192, d1121)': dict(channels=(32, 64, 128, 192), depths=(1, 1, 2, 1)),
    'M  (48-320, d2232)': dict(channels=(48, 96, 192, 320), depths=(2, 2, 3, 2)),  # production ~7M
    'L  (64-448, d2342)': dict(channels=(64, 128, 256, 448), depths=(2, 3, 4, 2)),
}

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('device:', device)


def load_g():
    if CACHE.exists():
        b = torch.load(CACHE); print('loaded cached G tensors'); return b['X'], b['y']
    df = pd.read_csv(DATA / 'train.csv')
    X = torch.empty(len(df), 1, IMG_H, IMG_W)
    for i, row in enumerate(df.itertuples(index=False)):
        g = np.array(Image.open(DATA / 'train' / row.id))[:, :, 1].astype(np.float32) / 255.0
        t = torch.from_numpy(g)[None, None]
        X[i] = F.interpolate(t, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)[0]
        if (i + 1) % 5000 == 0: print(f'  decoded {i+1}/{len(df)}')
    y = torch.tensor(df['label'].values - 1)
    torch.save({'X': X, 'y': y}, CACHE); return X, y


# ---- model (same primitives as the notebook's SignalNetV2) ----
class DropPath(nn.Module):
    def __init__(self, p=0.0): super().__init__(); self.p = p
    def forward(self, x):
        if self.p == 0.0 or not self.training: return x
        keep = 1 - self.p
        m = torch.empty((x.size(0),) + (1,)*(x.ndim-1), dtype=x.dtype, device=x.device).bernoulli_(keep)
        return x / keep * m

class SE(nn.Module):
    def __init__(self, c, r=8):
        super().__init__(); s = max(1, c // r); self.c = c
        self.fc = nn.Sequential(nn.Linear(c, s, bias=False), nn.SiLU(),
                                nn.Linear(s, c, bias=False), nn.Sigmoid())
    def forward(self, x): return x * self.fc(x.mean((2, 3))).view(x.size(0), self.c, 1, 1)

class BlurPool(nn.Module):
    def __init__(self, c):
        super().__init__(); self.c = c
        k = torch.tensor([1., 2., 1.]); k = (k[:, None] * k[None, :]); k = k / k.sum()
        self.register_buffer('k', k[None, None].repeat(c, 1, 1, 1))
    def forward(self, x): return F.conv2d(x, self.k, stride=2, padding=1, groups=self.c)

class CBA(nn.Module):
    def __init__(self, i, o, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(i, o, k, s, p, bias=False); self.bn = nn.BatchNorm2d(o); self.act = nn.SiLU(True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class Res(nn.Module):
    def __init__(self, c, dp=0.0):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm2d(c)
        self.se = SE(c); self.act = nn.SiLU(True); self.dp = DropPath(dp)
    def forward(self, x):
        o = self.act(self.b1(self.c1(x))); o = self.se(self.b2(self.c2(o)))
        return self.act(x + self.dp(o))

class Down(nn.Module):
    def __init__(self, i, o): super().__init__(); self.conv = CBA(i, o); self.pool = BlurPool(o)
    def forward(self, x): return self.pool(self.conv(x))

class Net(nn.Module):
    def __init__(self, channels, depths, nc=5, p=0.1):
        super().__init__()
        self.stem = CBA(1, channels[0])
        stages, prev = [], channels[0]
        for c, d in zip(channels, depths):
            layers = [Down(prev, c)] + [Res(c) for _ in range(d)]
            stages.append(nn.Sequential(*layers)); prev = c
        self.stages = nn.Sequential(*stages)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Linear(prev, prev), nn.SiLU(True), nn.Dropout(p), nn.Linear(prev, nc))
        for m in self.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None: nn.init.zeros_(m.bias)
        for m in self.modules():
            if isinstance(m, Res): nn.init.zeros_(m.b2.weight)
    def forward(self, x): return self.head(self.stages(self.stem(x)))


class EMA:
    def __init__(self, model, decay=0.995):
        self.decay = decay; self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def update(self, model):
        for s, m in zip(self.shadow.state_dict().values(), model.state_dict().values()):
            if s.dtype.is_floating_point: s.mul_(self.decay).add_(m, alpha=1 - self.decay)
            else: s.copy_(m)


class AugDS(Dataset):
    def __init__(self, X, y, mean, std, train, fmask=8, tmask=4):
        self.X, self.y, self.m, self.s, self.train, self.fm, self.tm = X, y, mean, std, train, fmask, tmask
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        x = self.X[i].clone()
        if self.train:
            c = 0.8 + 0.4 * torch.rand(1).item(); b = (torch.rand(1).item() - 0.5) * 0.1
            x = ((x - 0.5) * c + 0.5 + b).clamp_(0, 1)
            x.add_(torch.randn_like(x) * 0.02).clamp_(0, 1)
            H, W = x.shape[1], x.shape[2]
            f = int(torch.randint(0, self.fm + 1, (1,)).item()); f0 = int(torch.randint(0, max(1, H - f), (1,)).item()); x[:, f0:f0+f, :] = 0.0
            t = int(torch.randint(0, self.tm + 1, (1,)).item()); t0 = int(torch.randint(0, max(1, W - t), (1,)).item()); x[:, :, t0:t0+t] = 0.0
        return (x - self.m) / self.s, int(self.y[i])


def evaluate(model, Xva, yva):
    model.eval(); correct = 0
    with torch.no_grad():
        for s in range(0, len(Xva), 256):
            p = model(Xva[s:s+256].to(device)).argmax(1).cpu()
            correct += (p == yva[s:s+256]).sum().item()
    return 100 * correct / len(Xva)


def run_arm(name, cfg, X, y, tr, va):
    torch.manual_seed(SEED); np.random.seed(SEED)
    mean = X[tr].mean().item(); std = X[tr].std().item()
    gen = torch.Generator().manual_seed(SEED)
    tl = DataLoader(AugDS(X[tr], y[tr], mean, std, True), batch_size=BATCH,
                    shuffle=True, num_workers=0, drop_last=True, generator=gen)
    Xva = ((X[va] - mean) / std); yva = y[va]

    model = Net(cfg['channels'], cfg['depths']).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    ema = EMA(model, 0.995)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2.5e-3, total_steps=EPOCHS * len(tl), pct_start=0.25)

    best = 0.0
    for ep in range(EPOCHS):
        if ep == 3: ema = EMA(model, 0.995)
        model.train(); tc = torch.zeros((), device=device); tn = 0
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); out = model(xb); loss = crit(out, yb)
            loss.backward(); opt.step(); sched.step(); ema.update(model)
            with torch.no_grad(): tc += (out.argmax(1) == yb).sum()
            tn += yb.size(0)
        ta = 100 * tc.item() / tn
        rv = evaluate(model, Xva, yva); ev = evaluate(ema.shadow, Xva, yva)
        v = max(rv, ev); best = max(best, v)
        print(f'  [{name}] ep {ep+1:2d}/{EPOCHS}  train {ta:5.2f}  raw {rv:5.2f}  ema {ev:5.2f}  '
              f'-> val {v:5.2f} (best {best:5.2f})  gap {ta-v:+.1f}', flush=True)
    return best, nparams


def main():
    X, y = load_g()
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, stratify=y.numpy(), random_state=42)
    tr, va = torch.from_numpy(tr), torch.from_numpy(va)

    results = {}
    t0 = time.time()
    for name, cfg in CONFIGS.items():
        print(f'\n=== CAPACITY {name} ===')
        best, npar = run_arm(name, cfg, X, y, tr, va)
        results[name] = (best, npar)

    print('\n==================== A/B RESULT (capacity) ====================')
    base = results['M  (48-320, d2232)'][0]
    for name, (best, npar) in results.items():
        print(f'{name}  {npar/1e6:5.2f}M  best val: {best:5.2f}%   (Δ vs M: {best-base:+.2f} pp)')
    print(f'(elapsed {time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()

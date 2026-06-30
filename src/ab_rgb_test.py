"""
Controlled A/B #2: full RGB (3-channel) input vs single G channel.

Same architecture family, split, data order, augmentation RNG, optimizer,
schedule and seed. Only the input channels differ (so the stem conv differs in
shape — init cannot be bit-identical across arms, but everything else is).
"""
import time
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
SEED = 0
EPOCHS = 22
BATCH = 128
CACHE = Path('/tmp/ab_rgb_inputs.pt')

device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print('device:', device)


def build_inputs():
    if CACHE.exists():
        b = torch.load(CACHE)
        print('loaded cached inputs')
        return b['XG'], b['XR'], b['y']
    df = pd.read_csv(DATA / 'train.csv')
    N = len(df)
    XG = torch.empty(N, 1, IMG_H, IMG_W)
    XR = torch.empty(N, 3, IMG_H, IMG_W)
    for i, row in enumerate(df.itertuples(index=False)):
        arr = np.array(Image.open(DATA / 'train' / row.id))[:, :, :3].astype(np.float32) / 255.0
        rgb = torch.from_numpy(arr).permute(2, 0, 1)[None]      # [1,3,H,W]
        rgb = F.interpolate(rgb, size=(IMG_H, IMG_W), mode='bilinear', align_corners=False)[0]
        XR[i] = rgb
        XG[i] = rgb[1:2]                                         # G channel
        if (i + 1) % 3000 == 0:
            print(f'  decoded {i+1}/{N}')
    y = torch.tensor(df['label'].values - 1)
    torch.save({'XG': XG, 'XR': XR, 'y': y}, CACHE)
    return XG, XR, y


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
    def __init__(self, in_ch=1, ch=(32, 64, 128, 192), nc=5, p=0.3):
        super().__init__()
        self.stem = CBA(in_ch, ch[0])
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
    C = X.shape[1]
    mean = X[tr].mean(dim=(0, 2, 3)).view(C, 1, 1)
    std = X[tr].std(dim=(0, 2, 3)).view(C, 1, 1)
    gen = torch.Generator().manual_seed(SEED)
    tl = DataLoader(AugDS(X[tr], y[tr], mean, std, True), batch_size=BATCH,
                    shuffle=True, num_workers=0, drop_last=True, generator=gen)
    Xva = ((X[va] - mean) / std); yva = y[va]

    model = Net(in_ch=C).to(device)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=EPOCHS * len(tl), pct_start=0.15)

    best = 0.0
    for ep in range(EPOCHS):
        model.train()
        for xb, yb in tl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(); loss = crit(model(xb), yb); loss.backward(); opt.step(); sched.step()
        model.eval(); correct = 0
        with torch.no_grad():
            for s in range(0, len(Xva), 512):
                p = model(Xva[s:s+512].to(device)).argmax(1).cpu()
                correct += (p == yva[s:s+512]).sum().item()
        acc = 100 * correct / len(Xva); best = max(best, acc)
        print(f'  [{name}] ep {ep+1:2d}/{EPOCHS}  val {acc:5.2f}  (best {best:5.2f})', flush=True)
    return best


def main():
    XG, XR, y = build_inputs()
    print(f'dataset: {len(y)} imgs | G {tuple(XG.shape[1:])} | RGB {tuple(XR.shape[1:])}')
    idx = np.arange(len(y))
    tr, va = train_test_split(idx, test_size=0.2, stratify=y.numpy(), random_state=42)
    tr, va = torch.from_numpy(tr), torch.from_numpy(va)

    t0 = time.time()
    print('\n=== ARM A: G channel (1ch) ===')
    bg = run_arm('G  ', XG, y, tr, va)
    print('\n=== ARM B: full RGB (3ch) ===')
    br = run_arm('RGB', XR, y, tr, va)

    print('\n==================== A/B RESULT ====================')
    print(f'G channel (1ch)  best val: {bg:.2f}%')
    print(f'full RGB  (3ch)  best val: {br:.2f}%')
    print(f'delta (RGB - G):           {br - bg:+.2f} pp')
    print(f'(elapsed {time.time()-t0:.0f}s)')


if __name__ == '__main__':
    main()

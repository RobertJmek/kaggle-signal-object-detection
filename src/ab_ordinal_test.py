"""
Phase 1 A/B (path-to-80 campaign): does modeling the label as an ORDINAL COUNT beat nominal softmax?

Three arms, identical from-scratch MBConv backbone (B0-layout, the Run 24 architecture), identical
data/aug/optimizer/seed — ONLY the head + loss + decode differ:
  A. SOFTMAX     : 5 logits, CrossEntropy(label_smoothing=0.1), pred = argmax+1
  B. CORN        : 4 logits (ordinal "y>t" thresholds), CORN conditional-binary loss, rank decode
  C. REGRESSION  : 1 logit, SmoothL1 to count-1, pred = round(clamp(.,0,4))+1

Metric = exact-count accuracy (the Kaggle metric). Also reports ±1 accuracy.
Single 80/20 stratified split (seed 42), full 15.5k images, RAM-cached uint8, count-preserving GPU
aug (translation + contrast + noise, NO masking — masking can erase objects and corrupt the count).

Gate: promote the best of {CORN, REGRESSION} only if it beats SOFTMAX by >= +0.7pp exact acc.

Run from src/:  python3 ab_ordinal_test.py
"""
import math, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

DATA = Path('../data/raw')
S = 160               # input resolution (square; fast on MPS, preserves enough spatial detail to count)
EPOCHS = 36
BATCH = 96
MAX_LR = 1.2e-3
WD = 2e-3
SEED = 42
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


# ----------------------------- model (from-scratch MBConv, == notebook EffNet) -----------------------------
class DropPath(nn.Module):
    def __init__(s, p=0.0): super().__init__(); s.p = p
    def forward(s, x):
        if s.p == 0.0 or not s.training: return x
        k = 1 - s.p
        m = torch.empty((x.size(0), 1, 1, 1), dtype=x.dtype, device=x.device).bernoulli_(k)
        return x / k * m


class MBConv(nn.Module):
    def __init__(s, cin, cout, k=3, st=1, t=6, se=0.25, dp=0.0):
        super().__init__()
        cm = cin * t; s.use_res = (st == 1 and cin == cout); L = []
        if t != 1: L += [nn.Conv2d(cin, cm, 1, bias=False), nn.BatchNorm2d(cm), nn.SiLU(True)]
        L += [nn.Conv2d(cm, cm, k, st, k // 2, groups=cm, bias=False), nn.BatchNorm2d(cm), nn.SiLU(True)]
        s.conv = nn.Sequential(*L); sc = max(1, int(cin * se))
        s.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cm, sc, 1), nn.SiLU(True),
                             nn.Conv2d(sc, cm, 1), nn.Sigmoid())
        s.proj = nn.Sequential(nn.Conv2d(cm, cout, 1, bias=False), nn.BatchNorm2d(cout)); s.dp = DropPath(dp)
        if s.use_res: nn.init.zeros_(s.proj[1].weight)
    def forward(s, x):
        o = s.conv(x); o = o * s.se(o); o = s.proj(o); return x + s.dp(o) if s.use_res else o


class EffNet(nn.Module):
    cfg = [(1, 16, 3, 1, 1), (6, 24, 3, 2, 2), (6, 40, 5, 2, 2), (6, 80, 3, 2, 3),
           (6, 112, 5, 1, 3), (6, 192, 5, 2, 4), (6, 320, 3, 1, 1)]
    def __init__(s, n_out=5, dropout=0.3, drop_path=0.15, width=1.0, depth_mult=1.0):
        super().__init__()
        ch = lambda c: int(c * width); rep = lambda r: int(math.ceil(r * depth_mult))
        s.stem = nn.Sequential(nn.Conv2d(3, ch(32), 3, 2, 1, bias=False), nn.BatchNorm2d(ch(32)), nn.SiLU(True))
        blocks = []; cin = ch(32); tot = sum(rep(r) for *_, r in s.cfg); bi = 0
        for t, co, k, st, r in s.cfg:
            co = ch(co)
            for j in range(rep(r)):
                blocks.append(MBConv(cin, co, k, st if j == 0 else 1, t, dp=drop_path * bi / max(1, tot - 1)))
                cin = co; bi += 1
        s.blocks = nn.Sequential(*blocks)
        s.head = nn.Sequential(nn.Conv2d(cin, ch(1280), 1, bias=False), nn.BatchNorm2d(ch(1280)), nn.SiLU(True),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(ch(1280), n_out))
        for m in s.modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(s, x): return s.head(s.blocks(s.stem(x)))


class EMA:
    def __init__(s, m, d=0.999):
        import copy
        s.d = d; s.sh = copy.deepcopy(m).eval()
        [p.requires_grad_(False) for p in s.sh.parameters()]
    @torch.no_grad()
    def update(s, m):
        for a, b in zip(s.sh.state_dict().values(), m.state_dict().values()):
            if a.dtype.is_floating_point: a.mul_(s.d).add_(b, alpha=1 - s.d)
            else: a.copy_(b)


# ----------------------------- heads: loss + decode -----------------------------
def corn_loss(logits, y0):                      # logits [B,4], y0 [B] in 0..4
    loss = 0.0; nt = logits.shape[1]; used = 0
    for t in range(nt):
        mask = y0 >= t                          # conditional subset (CORN)
        if mask.sum() == 0: continue
        target = (y0[mask] > t).float()
        loss = loss + F.binary_cross_entropy_with_logits(logits[mask, t], target); used += 1
    return loss / max(1, used)


def corn_decode(logits):
    cum = torch.cumprod(torch.sigmoid(logits), dim=1)   # P(y>t) unconditional
    return (cum > 0.5).sum(1)                            # rank 0..4 -> count = rank+1


HEADS = {
    'softmax': dict(n_out=5,
                    loss=lambda out, y0: F.cross_entropy(out, y0, label_smoothing=0.1),
                    decode=lambda out: out.argmax(1)),
    'corn':    dict(n_out=4,
                    loss=lambda out, y0: corn_loss(out, y0),
                    decode=lambda out: corn_decode(out)),
    'reg':     dict(n_out=1,
                    loss=lambda out, y0: F.smooth_l1_loss(out.squeeze(1), y0.float()),
                    decode=lambda out: out.squeeze(1).round().clamp(0, 4).long()),
}


# ----------------------------- data -----------------------------
def cache():
    cf = Path(f'../models/ab_ordinal_cache_{S}.pt')
    if cf.exists():
        b = torch.load(cf); print('loaded cache'); return b['X'], b['y']
    df = pd.read_csv(DATA / 'train.csv')
    X = torch.empty(len(df), 3, S, S, dtype=torch.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        a = np.asarray(Image.open(DATA / 'train' / row.id).convert('RGB').resize((S, S)))
        X[i] = torch.from_numpy(a.copy()).permute(2, 0, 1)
        if (i + 1) % 3000 == 0: print(f'  cached {i+1}/{len(df)}')
    y = torch.tensor(df['label'].values - 1)
    cf.parent.mkdir(exist_ok=True); torch.save({'X': X, 'y': y}, cf)
    return X, y


def gpu_aug(x):                                 # count-preserving only: translation + contrast/bright + noise
    fs = int(torch.randint(-12, 13, (1,)).item()); ts = int(torch.randint(-40, 41, (1,)).item())
    x = torch.roll(x, shifts=(fs, ts), dims=(2, 3))
    if fs > 0: x[:, :, :fs, :] = 0
    elif fs < 0: x[:, :, fs:, :] = 0
    if ts > 0: x[:, :, :, :ts] = 0
    elif ts < 0: x[:, :, :, ts:] = 0
    B = x.size(0)
    c = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=x.device)
    b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.1
    sig = 0.01 + 0.05 * torch.rand(B, 1, 1, 1, device=x.device)
    x = (x - 0.5) * c + 0.5 + b + torch.randn_like(x) * sig
    return x.clamp_(0, 1)


def run_arm(name, X, y, tr, va, mean, std):
    h = HEADS[name]
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = EffNet(n_out=h['n_out']).to(device)
    ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    tr_t = torch.as_tensor(tr); steps = len(tr_t) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=EPOCHS * steps, pct_start=0.2)
    md = mean.to(device); sd = std.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f'\n[{name}] {n_params/1e6:.2f}M params | {len(tr_t)} train / {len(va)} val | {steps} batches/ep', flush=True)

    @torch.no_grad()
    def evaluate(m):
        m.eval(); ex = pm1 = 0
        for s0 in range(0, len(va), 256):
            bi = va[s0:s0 + 256]
            xb = (X[bi].to(device).float() / 255. - md) / sd
            pred = h['decode'](m(xb)).cpu()
            tgt = y[bi]
            ex += (pred == tgt).sum().item(); pm1 += (pred - tgt).abs().le(1).sum().item()
        return 100 * ex / len(va), 100 * pm1 / len(va)

    best_ex = 0.0; best_pm1 = 0.0; t0 = time.time()
    for ep in range(EPOCHS):
        if ep == 3: ema = EMA(model, 0.999)
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for s0 in range(0, steps * BATCH, BATCH):
            bi = perm[s0:s0 + BATCH]
            xb = (gpu_aug(X[bi].to(device).float() / 255.) - md) / sd
            yb = y[bi].to(device)
            opt.zero_grad(set_to_none=True)
            out = model(xb); loss = h['loss'](out, yb)
            loss.backward(); opt.step(); sched.step(); ema.update(model)
        rex, rp1 = evaluate(model); eex, ep1 = evaluate(ema.sh)
        ex = max(rex, eex); pm1 = ep1 if eex >= rex else rp1
        tag = 'EMA' if eex >= rex else 'raw'
        if ex > best_ex: best_ex = ex; best_pm1 = pm1
        if (ep + 1) % 4 == 0 or ep < 2:
            print(f'  [{name}] ep{ep+1:2d}/{EPOCHS} exact {ex:5.2f} (best {best_ex:5.2f}) +/-1 {pm1:5.2f} [{tag}]', flush=True)
    print(f'[{name}] BEST exact {best_ex:.2f}%  (+/-1 {best_pm1:.2f}%)  [{time.time()-t0:.0f}s]', flush=True)
    return best_ex, best_pm1


def main():
    from sklearn.model_selection import StratifiedKFold
    print(f'device={device}  S={S}  epochs={EPOCHS}  batch={BATCH}')
    X, y = cache()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))   # 80/20
    va = torch.as_tensor(va)
    px = X[torch.as_tensor(tr)].float() / 255.
    mean = px.mean(dim=(0, 2, 3), keepdim=True)[0]; std = px.std(dim=(0, 2, 3), keepdim=True)[0]
    del px
    print(f'mean={mean.flatten().tolist()} std={std.flatten().tolist()}')

    res = {}
    for name in ['softmax', 'corn', 'reg']:
        res[name] = run_arm(name, X, y, tr, va, mean, std)

    print('\n===== PHASE 1 RESULT (exact-count accuracy) =====')
    base = res['softmax'][0]
    for name in ['softmax', 'corn', 'reg']:
        ex, pm1 = res[name]
        print(f'  {name:8s} exact {ex:5.2f}%  +/-1 {pm1:5.2f}%  (Δ vs softmax {ex-base:+.2f})')
    best_ord = max(res['corn'][0], res['reg'][0])
    print(f'\nGATE (+0.7pp): best ordinal {best_ord:.2f} vs softmax {base:.2f} -> '
          f'{"PROMOTE" if best_ord >= base + 0.7 else "no clear win; keep softmax"}')


if __name__ == '__main__':
    main()

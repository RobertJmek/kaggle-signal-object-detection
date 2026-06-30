"""
Unified A/B harness for the path-to-80 campaign — Phases 2 (augmentation) and 3 (oriented stem).

Generalizes src/ab_ordinal_test.py with two extra knobs, holding everything else identical:
  --head   softmax|corn|reg       (use the Phase-1 winner)
  --stem   plain|gabor-fixed|gabor-learn   (Phase 3)
  --aug    safe|mask|heavy-safe            (Phase 2)
  --arms   comma list of "label:stem:aug" specs to compare in one run

Examples:
  # Phase 2 (aug A/B), winning head = corn:
  python3 ab_campaign.py --head corn --arms "SAFE:plain:safe,MASK:plain:mask,HEAVYSAFE:plain:heavy-safe"
  # Phase 3 (stem A/B), winning head = corn, winning aug = safe:
  python3 ab_campaign.py --head corn --arms "BASE:plain:safe,GABOR-FIX:gabor-fixed:safe,GABOR-LRN:gabor-learn:safe"

Metric = exact-count accuracy (+/-1 reported). Single 80/20 stratified split (seed 42), full 15.5k,
RAM-cached uint8. Gate: Phase 2 pick highest exact; Phase 3 promote Gabor only if >= +1.0pp vs BASE.
"""
import argparse, math, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F

from ab_ordinal_test import (DropPath, MBConv, EMA, HEADS, S, DATA,
                             EPOCHS, BATCH, MAX_LR, WD, SEED, device)


# ----------------------------- oriented (Gabor) stem — Phase 3, primitives only -----------------------------
class OrientedStem(nn.Module):
    """Multi-orientation Gabor/line-detector front-end (legal: nn.Conv2d, analytic init, self-contained).
    Replaces EffNet's stem; outputs out_ch channels at stride 2, ends BN+SiLU (same interface)."""
    def __init__(s, cin=3, n_orient=8, scales=(2.0, 4.0), ksize=7, stride=2, out_ch=32,
                 learnable=True, mix_plain=True):
        super().__init__()
        s.n = n_orient * len(scales)
        s.to_gray = nn.Conv2d(cin, 1, 1, bias=False)
        nn.init.constant_(s.to_gray.weight, 1.0 / cin)
        gabor = s._build_gabor(n_orient, scales, ksize)
        s.gabor = nn.Conv2d(1, s.n, ksize, stride, ksize // 2, bias=False)
        with torch.no_grad():
            s.gabor.weight.copy_(gabor)
        s.gabor.weight.requires_grad_(learnable)
        s.mix_plain = mix_plain
        if mix_plain:
            s.plain = nn.Conv2d(cin, out_ch - s.n, ksize, stride, ksize // 2, bias=False)
            nn.init.kaiming_normal_(s.plain.weight, mode='fan_out', nonlinearity='relu')
        fuse_in = s.n + (out_ch - s.n if mix_plain else 0)
        s.fuse = nn.Conv2d(fuse_in, out_ch, 1, bias=False)
        s.bn = nn.BatchNorm2d(out_ch); s.act = nn.SiLU(True)

    @staticmethod
    def _build_gabor(n_orient, scales, k):
        half = k // 2
        ys, xs = torch.meshgrid(torch.arange(-half, half + 1).float(),
                                torch.arange(-half, half + 1).float(), indexing='ij')
        filters = []
        for lam in scales:
            sigma = 0.56 * lam; gamma = 0.5
            for o in range(n_orient):
                theta = math.pi * o / n_orient
                xr = xs * math.cos(theta) + ys * math.sin(theta)
                yr = -xs * math.sin(theta) + ys * math.cos(theta)
                env = torch.exp(-(xr ** 2 + (gamma * yr) ** 2) / (2 * sigma ** 2))
                g = env * torch.cos(2 * math.pi * xr / lam)
                g = g - g.mean(); g = g / (g.norm() + 1e-6)
                filters.append(g)
        return torch.stack(filters).unsqueeze(1)

    def forward(s, x):
        g = s.gabor(s.to_gray(x))
        if s.mix_plain:
            g = torch.cat([g, s.plain(x)], dim=1)
        return s.act(s.bn(s.fuse(g)))


class EffNet(nn.Module):
    cfg = [(1, 16, 3, 1, 1), (6, 24, 3, 2, 2), (6, 40, 5, 2, 2), (6, 80, 3, 2, 3),
           (6, 112, 5, 1, 3), (6, 192, 5, 2, 4), (6, 320, 3, 1, 1)]
    def __init__(s, n_out=5, dropout=0.3, drop_path=0.15, width=1.0, depth_mult=1.0, stem='plain'):
        super().__init__()
        ch = lambda c: int(c * width); rep = lambda r: int(math.ceil(r * depth_mult))
        if stem == 'plain':
            s.stem = nn.Sequential(nn.Conv2d(3, ch(32), 3, 2, 1, bias=False), nn.BatchNorm2d(ch(32)), nn.SiLU(True))
        else:
            s.stem = OrientedStem(out_ch=ch(32), learnable=(stem == 'gabor-learn'), mix_plain=True)
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


# ----------------------------- augmentation variants — Phase 2 -----------------------------
def make_aug(mode):
    def aug(x):
        fs = int(torch.randint(-12, 13, (1,)).item()); ts = int(torch.randint(-40, 41, (1,)).item())
        x = torch.roll(x, shifts=(fs, ts), dims=(2, 3))
        if fs > 0: x[:, :, :fs, :] = 0
        elif fs < 0: x[:, :, fs:, :] = 0
        if ts > 0: x[:, :, :, :ts] = 0
        elif ts < 0: x[:, :, :, ts:] = 0
        B = x.size(0)
        cmax = 0.4 if mode != 'heavy-safe' else 0.6
        nhi = 0.05 if mode != 'heavy-safe' else 0.09
        c = (1 - cmax / 2) + cmax * torch.rand(B, 1, 1, 1, device=x.device)
        b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.1
        sig = 0.01 + nhi * torch.rand(B, 1, 1, 1, device=x.device)
        x = (x - 0.5) * c + 0.5 + b + torch.randn_like(x) * sig
        if mode == 'mask':                       # SpecAugment — tests whether masking erases objects
            H, W = x.shape[2], x.shape[3]
            fm = int(torch.randint(0, 17, (1,)).item())
            if fm > 0:
                f0 = int(torch.randint(0, max(1, H - fm), (1,)).item()); x[:, :, f0:f0 + fm, :] = 0
            tm = int(torch.randint(0, 25, (1,)).item())
            if tm > 0:
                t0 = int(torch.randint(0, max(1, W - tm), (1,)).item()); x[:, :, :, t0:t0 + tm] = 0
        return x.clamp_(0, 1)
    return aug


def cache():
    cf = Path(f'../models/ab_ordinal_cache_{S}.pt')           # reuse Phase-1 cache
    b = torch.load(cf); return b['X'], b['y']


def run_arm(label, head, stem, aug_mode, X, y, tr, va, mean, std):
    h = HEADS[head]; aug = make_aug(aug_mode)
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = EffNet(n_out=h['n_out'], stem=stem).to(device)
    ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    tr_t = torch.as_tensor(tr); steps = len(tr_t) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=EPOCHS * steps, pct_start=0.2)
    md = mean.to(device); sd = std.to(device)
    print(f'\n[{label}] head={head} stem={stem} aug={aug_mode} | '
          f'{sum(p.numel() for p in model.parameters())/1e6:.2f}M | {steps} batches/ep', flush=True)

    @torch.no_grad()
    def evaluate(m):
        m.eval(); ex = pm1 = 0
        for s0 in range(0, len(va), 256):
            bi = va[s0:s0 + 256]
            xb = (X[bi].to(device).float() / 255. - md) / sd
            pred = h['decode'](m(xb)).cpu(); tgt = y[bi]
            ex += (pred == tgt).sum().item(); pm1 += (pred - tgt).abs().le(1).sum().item()
        return 100 * ex / len(va), 100 * pm1 / len(va)

    best_ex = best_pm1 = 0.0; t0 = time.time()
    for ep in range(EPOCHS):
        if ep == 3: ema = EMA(model, 0.999)
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for s0 in range(0, steps * BATCH, BATCH):
            bi = perm[s0:s0 + BATCH]
            xb = (aug(X[bi].to(device).float() / 255.) - md) / sd
            yb = y[bi].to(device)
            opt.zero_grad(set_to_none=True)
            loss = h['loss'](model(xb), yb)
            loss.backward(); opt.step(); sched.step(); ema.update(model)
        rex, _ = evaluate(model); eex, ep1 = evaluate(ema.sh)
        ex = max(rex, eex); pm1 = ep1 if eex >= rex else _
        if ex > best_ex: best_ex, best_pm1 = ex, pm1
        if (ep + 1) % 6 == 0:
            print(f'  [{label}] ep{ep+1:2d}/{EPOCHS} exact {ex:5.2f} (best {best_ex:5.2f})', flush=True)
    print(f'[{label}] BEST exact {best_ex:.2f}%  (+/-1 {best_pm1:.2f}%)  [{time.time()-t0:.0f}s]', flush=True)
    return best_ex, best_pm1


def main():
    from sklearn.model_selection import StratifiedKFold
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', default='corn')
    ap.add_argument('--arms', default='SAFE:plain:safe,MASK:plain:mask')
    a = ap.parse_args()
    print(f'device={device} S={S} epochs={EPOCHS} head={a.head}')
    X, y = cache()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
    va = torch.as_tensor(va)
    px = X[torch.as_tensor(tr)].float() / 255.
    mean = px.mean(dim=(0, 2, 3), keepdim=True)[0]; std = px.std(dim=(0, 2, 3), keepdim=True)[0]; del px

    arms = [spec.split(':') for spec in a.arms.split(',')]
    res = {}
    for label, stem, aug_mode in arms:
        res[label] = run_arm(label, a.head, stem, aug_mode, X, y, tr, va, mean, std)
    print('\n===== CAMPAIGN A/B RESULT (exact-count accuracy) =====')
    base = list(res.values())[0][0]; base_lbl = list(res.keys())[0]
    for label, (ex, pm1) in res.items():
        print(f'  {label:12s} exact {ex:5.2f}%  +/-1 {pm1:5.2f}%  (Δ vs {base_lbl} {ex-base:+.2f})')


if __name__ == '__main__':
    main()

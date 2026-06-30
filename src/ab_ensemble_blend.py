"""
MULTI-ARCH ENSEMBLE BLEND (path-to-80, Phase 5) — does a distinct from-scratch architecture blend
with the MBConv EffNet to beat the best single model?

Aspect / resolution / capacity are all tuned (Exp 41/42). The remaining lever is ensembling a net with a
DIFFERENT inductive bias so its errors are less correlated with EffNet's. Second arch = DilatedCounter:
plain 3x3 residual blocks with growing DILATION (no depthwise, no SE, no inverted-bottleneck) — a wide
multi-scale receptive field suited to counting object SPACING, structurally unlike MBConv. Legal: built
from nn.Conv2d/BN/ReLU primitives, random init.

Trains both at native 256x110 on ONE StratifiedKFold(5,seed42) split, snapshots each model's val softmax
probs at its best epoch, reports standalone EffNet / standalone DilatedCounter / 50-50 blend exact acc.
Gate: blend >= best_single + 0.5 -> ensemble is worth the 2x deployment cost (add as a fold member).

Run from src/:  python3 ab_ensemble_blend.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from ab_aspect import build_cache
from ab_ordinal_test import EffNet, EMA, BATCH, MAX_LR, WD, SEED, device, gpu_aug

EPOCHS = 32
H, W = 256, 110


# ---------------- distinct arch: dilated-residual counter (no depthwise / SE / MBConv) ----------------
class DilBlock(nn.Module):
    def __init__(s, cin, cout, stride=1, dil=1):
        super().__init__()
        s.c1 = nn.Conv2d(cin, cout, 3, stride, dil, dilation=dil, bias=False); s.b1 = nn.BatchNorm2d(cout)
        s.c2 = nn.Conv2d(cout, cout, 3, 1, dil, dilation=dil, bias=False);      s.b2 = nn.BatchNorm2d(cout)
        s.short = (nn.Sequential() if stride == 1 and cin == cout else
                   nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        nn.init.zeros_(s.b2.weight)                       # identity init (Exp 16 trick)
    def forward(s, x):
        o = F.relu(s.b1(s.c1(x)), True); o = s.b2(s.c2(o))
        return F.relu(o + s.short(x), True)


class DilatedCounter(nn.Module):
    def __init__(s, n_out=5, width=48, drop=0.3):
        super().__init__()
        c = width
        s.stem = nn.Sequential(nn.Conv2d(3, c, 3, 2, 1, bias=False), nn.BatchNorm2d(c), nn.ReLU(True))
        s.stage1 = nn.Sequential(DilBlock(c, c, 1, 1), DilBlock(c, c, 1, 1))
        s.stage2 = nn.Sequential(DilBlock(c, 2*c, 2, 1), DilBlock(2*c, 2*c, 1, 2))      # dilation 2
        s.stage3 = nn.Sequential(DilBlock(2*c, 4*c, 2, 1), DilBlock(4*c, 4*c, 1, 3))    # dilation 3
        s.stage4 = nn.Sequential(DilBlock(4*c, 8*c, 2, 1), DilBlock(8*c, 8*c, 1, 2))
        s.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(drop), nn.Linear(8*c, n_out))
        for m in s.modules():
            if isinstance(m, nn.Conv2d): nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(s, x):
        return s.head(s.stage4(s.stage3(s.stage2(s.stage1(s.stem(x))))))


def train_capture(name, model, X, y, tr, va, md, sd):
    """train; return (best_acc, val_probs@best) — probs are softmax at the best-val epoch."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = model.to(device); ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    tr_t = torch.as_tensor(tr); steps = len(tr_t) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=EPOCHS*steps, pct_start=0.2)
    print(f'\n[{name}] {sum(p.numel() for p in model.parameters())/1e6:.2f}M | {steps} batches/ep', flush=True)

    @torch.no_grad()
    def probs_of(m):
        m.eval(); out = []
        for s0 in range(0, len(va), 256):
            bi = va[s0:s0+256]
            xb = (X[bi].to(device).float()/255. - md)/sd
            out.append(F.softmax(m(xb), 1).cpu())
        return torch.cat(out)

    best = 0.0; best_probs = None; t0 = time.time()
    yv = y[va]
    for ep in range(EPOCHS):
        if ep == 3: ema = EMA(model, 0.999)
        model.train(); perm = tr_t[torch.randperm(len(tr_t))]
        for s0 in range(0, steps*BATCH, BATCH):
            bi = perm[s0:s0+BATCH]
            xb = (gpu_aug(X[bi].to(device).float()/255.) - md)/sd
            yb = y[bi].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            loss.backward(); opt.step(); sched.step(); ema.update(model)
        for m in (model, ema.sh):
            p = probs_of(m); acc = 100*(p.argmax(1) == yv).float().mean().item()
            if acc > best: best = acc; best_probs = p
        if (ep+1) % 8 == 0:
            print(f'  [{name}] ep{ep+1:2d}/{EPOCHS} best {best:5.2f}', flush=True)
    print(f'[{name}] BEST {best:.2f}%  [{time.time()-t0:.0f}s]', flush=True)
    return best, best_probs


def main():
    print(f'ensemble-blend @ native {H}x{W}', flush=True)
    X, y = build_cache('NAT_256x110', H, W)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
    va = torch.as_tensor(va)
    px = X[torch.as_tensor(tr)].float()/255.
    mean = px.mean(dim=(0,2,3), keepdim=True)[0]; std = px.std(dim=(0,2,3), keepdim=True)[0]; del px
    md = mean.to(device); sd = std.to(device); yv = y[va]

    a_eff, p_eff = train_capture('EffNet_w1.3', EffNet(5, width=1.3, depth_mult=1.2), X, y, tr, va, md, sd)
    a_dil, p_dil = train_capture('DilatedCounter', DilatedCounter(5), X, y, tr, va, md, sd)

    blend = 100*((p_eff + p_dil).argmax(1) == yv).float().mean().item()
    corr = torch.corrcoef(torch.stack([p_eff.argmax(1).float(), p_dil.argmax(1).float()]))[0, 1].item()
    best_single = max(a_eff, a_dil)
    print('\n===== ENSEMBLE BLEND (exact-count acc) =====', flush=True)
    print(f'  EffNet_w1.3      {a_eff:5.2f}%', flush=True)
    print(f'  DilatedCounter   {a_dil:5.2f}%', flush=True)
    print(f'  50-50 blend      {blend:5.2f}%   (best single {best_single:.2f}, Δ {blend-best_single:+.2f})', flush=True)
    print(f'  pred correlation {corr:.3f}  (lower = more complementary)', flush=True)
    if blend >= best_single + 0.5:
        print('  VERDICT: blend HELPS -> add DilatedCounter as an ensemble member in deployment.', flush=True)
    else:
        print('  VERDICT: blend does not clear +0.5 -> not worth 2x cost; rely on folds + pseudo.', flush=True)


if __name__ == '__main__':
    main()

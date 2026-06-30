"""
Phase 6 A/B (path-to-80): does an AUXILIARY count-regression head sharpen the softmax features?

Phase 1 (Exp 33) showed *replacing* the softmax decode with ordinal/regression hurts exact accuracy
(proximity losses blur the adjacent-count boundary). But the count signal might still help as an
AUXILIARY task: keep softmax as the primary head + decode, and ADD a count-regression head on shared
features with a small weight LAMBDA. The aux loss pressures the backbone to encode count precisely
without touching the (winning) argmax decode.

Arms (identical B0-MBConv backbone @160 RGB, full 15.5k, split seed 42, 36 ep, count-preserving aug):
  LAMBDA=0.0  -> pure softmax (== Exp 33 baseline 67.68, sanity)
  LAMBDA=0.3  -> softmax + 0.3 * SmoothL1(count)
  LAMBDA=0.6  -> softmax + 0.6 * SmoothL1(count)
Metric: exact-count accuracy (argmax of the softmax head). Gate: promote best aux >= baseline + 0.7pp.

Run from src/:  python3 ab_multitask.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from ab_ordinal_test import (DropPath, MBConv, EMA, S, EPOCHS, BATCH, MAX_LR, WD, SEED,
                             device, cache, gpu_aug)


class MultiHeadNet(nn.Module):
    """EffNet B0 backbone with a shared trunk -> two heads: softmax(5) + count-regression(1)."""
    cfg = [(1, 16, 3, 1, 1), (6, 24, 3, 2, 2), (6, 40, 5, 2, 2), (6, 80, 3, 2, 3),
           (6, 112, 5, 1, 3), (6, 192, 5, 2, 4), (6, 320, 3, 1, 1)]
    def __init__(s, dropout=0.3, drop_path=0.15, width=1.0, depth_mult=1.0):
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
        s.trunk = nn.Sequential(nn.Conv2d(cin, ch(1280), 1, bias=False), nn.BatchNorm2d(ch(1280)), nn.SiLU(True),
                                nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout))
        s.cls = nn.Linear(ch(1280), 5)        # primary: softmax classification (decode = argmax)
        s.reg = nn.Linear(ch(1280), 1)        # auxiliary: count regression
        for m in s.modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    def forward(s, x):
        f = s.trunk(s.blocks(s.stem(x)))
        return s.cls(f), s.reg(f).squeeze(1)


def run_arm(lam, X, y, tr, va, mean, std):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = MultiHeadNet().to(device); ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    tr_t = torch.as_tensor(tr); steps = len(tr_t) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=EPOCHS * steps, pct_start=0.2)
    md = mean.to(device); sd = std.to(device)
    print(f'\n[lam={lam}] {sum(p.numel() for p in model.parameters())/1e6:.2f}M | {steps} batches/ep', flush=True)

    @torch.no_grad()
    def evaluate(m):
        m.eval(); ex = 0
        for s0 in range(0, len(va), 256):
            bi = va[s0:s0 + 256]
            xb = (X[bi].to(device).float() / 255. - md) / sd
            cls, _ = m(xb)
            ex += (cls.argmax(1).cpu() == y[bi]).sum().item()
        return 100 * ex / len(va)

    best = 0.0; t0 = time.time()
    for ep in range(EPOCHS):
        if ep == 3: ema = EMA(model, 0.999)
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for s0 in range(0, steps * BATCH, BATCH):
            bi = perm[s0:s0 + BATCH]
            xb = (gpu_aug(X[bi].to(device).float() / 255.) - md) / sd
            yb = y[bi].to(device)
            opt.zero_grad(set_to_none=True)
            cls, reg = model(xb)
            loss = F.cross_entropy(cls, yb, label_smoothing=0.1)
            if lam > 0: loss = loss + lam * F.smooth_l1_loss(reg, yb.float())
            loss.backward(); opt.step(); sched.step(); ema.update(model)
        acc = max(evaluate(model), evaluate(ema.sh))
        if acc > best: best = acc
        if (ep + 1) % 6 == 0:
            print(f'  [lam={lam}] ep{ep+1:2d}/{EPOCHS} exact {acc:5.2f} (best {best:5.2f})', flush=True)
    print(f'[lam={lam}] BEST exact {best:.2f}%  [{time.time()-t0:.0f}s]', flush=True)
    return best


def main():
    from sklearn.model_selection import StratifiedKFold
    print(f'device={device} S={S} epochs={EPOCHS} (Phase 6 multi-task aux)')
    X, y = cache()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
    va = torch.as_tensor(va)
    px = X[torch.as_tensor(tr)].float() / 255.
    mean = px.mean(dim=(0, 2, 3), keepdim=True)[0]; std = px.std(dim=(0, 2, 3), keepdim=True)[0]; del px
    res = {lam: run_arm(lam, X, y, tr, va, mean, std) for lam in (0.0, 0.3, 0.6)}
    print('\n===== PHASE 6 RESULT (exact-count acc) =====')
    base = res[0.0]
    for lam, a in res.items():
        print(f'  lam={lam}: {a:5.2f}%  (Δ vs softmax {a-base:+.2f})')
    best_aux = max(res[0.3], res[0.6])
    print(f'GATE (+0.7): best aux {best_aux:.2f} vs {base:.2f} -> '
          f'{"PROMOTE" if best_aux >= base + 0.7 else "no win; keep plain softmax"}')


if __name__ == '__main__':
    main()

"""
CAPACITY @ NATIVE ASPECT (path-to-80, follow-up to Exp 41/42).

Capacity was ruled "exhausted" in Exp 21/23/26 — but ALL of those were on SQUARE distorted inputs where
the wall was overfitting. Native aspect (Exp 41, +3pp) reset the regime: the train/val gap is now NEGATIVE
(val > train, healthy — not overfitting). So the old "more capacity just overfits" conclusion is suspect.

This A/Bs model WIDTH at fixed native aspect 256x110 (the efficient resolution from the Exp 42 sweep):
  BASE_w1.0   : width=1.0 depth=1.0  (~4M, the proxy baseline)
  WIDE_w1.3   : width=1.3 depth=1.2  (~10M, beyond the 7.83M deployment)
If WIDE beats BASE by >= +0.7pp exact acc, bump CONFIG width/depth in run27_vast.py for the next run.
Same data/aug/EMA/OneCycle/seed as ab_aspect; only the model size differs.

Run from src/:  python3 ab_capacity_native.py
"""
import time
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from ab_aspect import build_cache
from ab_ordinal_test import EffNet, EMA, BATCH, MAX_LR, WD, SEED, device, gpu_aug

EPOCHS = 32
H, W = 256, 110                      # native aspect (H/W=2.327), efficient resolution from Exp 42 sweep
ARMS = {'BASE_w1.0': dict(width=1.0, depth_mult=1.0),
        'WIDE_w1.3': dict(width=1.3, depth_mult=1.2)}


def run_arm(tag, cfg, X, y, tr, va, mean, std):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = EffNet(5, **cfg).to(device); ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    tr_t = torch.as_tensor(tr); steps = len(tr_t) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=EPOCHS * steps, pct_start=0.2)
    md = mean.to(device); sd = std.to(device)
    print(f'\n[{tag}] {H}x{W} | {sum(p.numel() for p in model.parameters())/1e6:.2f}M | {steps} batches/ep', flush=True)

    @torch.no_grad()
    def evaluate(m):
        m.eval(); ex = 0
        for s0 in range(0, len(va), 256):
            bi = va[s0:s0+256]
            xb = (X[bi].to(device).float()/255. - md)/sd
            ex += (m(xb).argmax(1).cpu() == y[bi]).sum().item()
        return 100*ex/len(va)

    best = 0.0; t0 = time.time()
    for ep in range(EPOCHS):
        if ep == 3: ema = EMA(model, 0.999)
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for s0 in range(0, steps*BATCH, BATCH):
            bi = perm[s0:s0+BATCH]
            xb = (gpu_aug(X[bi].to(device).float()/255.) - md)/sd
            yb = y[bi].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            loss.backward(); opt.step(); sched.step(); ema.update(model)
        acc = max(evaluate(model), evaluate(ema.sh))
        if acc > best: best = acc
        if (ep+1) % 8 == 0:
            print(f'  [{tag}] ep{ep+1:2d}/{EPOCHS} exact {acc:5.2f} (best {best:5.2f})', flush=True)
    print(f'[{tag}] BEST exact {best:.2f}%  [{time.time()-t0:.0f}s]', flush=True)
    return best


def main():
    print(f'capacity@native {H}x{W} arms={list(ARMS)}', flush=True)
    X, y = build_cache('NAT_256x110', H, W)            # reuse the sweep's cache
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
    va = torch.as_tensor(va)
    px = X[torch.as_tensor(tr)].float()/255.
    mean = px.mean(dim=(0,2,3), keepdim=True)[0]; std = px.std(dim=(0,2,3), keepdim=True)[0]; del px
    res = {t: run_arm(t, c, X, y, tr, va, mean, std) for t, c in ARMS.items()}
    print('\n===== CAPACITY @ NATIVE ASPECT (exact-count acc) =====')
    for t, a in res.items(): print(f'  {t:<12} {a:5.2f}%', flush=True)
    d = res['WIDE_w1.3'] - res['BASE_w1.0']
    print(f'  Δ (WIDE - BASE) = {d:+.2f}', flush=True)
    if d >= 0.7:
        print('  VERDICT: capacity HELPS at native aspect -> bump CONFIG width=1.3/depth=1.2 for next run.', flush=True)
    elif d <= -0.7:
        print('  VERDICT: bigger overfits even at native aspect -> keep width=1.1.', flush=True)
    else:
        print('  VERDICT: capacity neutral (|Δ|<0.7) -> keep current width; spend budget elsewhere.', flush=True)


if __name__ == '__main__':
    main()

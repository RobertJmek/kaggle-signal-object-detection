"""
Resolution A/B (path-to-82) — does MORE input resolution help the COUNTING task?

Counting thin 1-2px oriented lines should benefit from more pixels separating close/overlapping
objects. All structural levers failed (ordinal/stem/masking); resolution is a cheap, untested,
non-architectural lever that — if it helps on the MPS proxy — justifies pushing the Colab run from
256 to 288/320. Same EffNet B0-MBConv, softmax head, count-preserving gpu_aug, EMA, OneCycle, and the
SAME StratifiedKFold(5, seed42) first split as Exp 33/35 so the S=160 SAFE baseline (67.68%) is the
direct reference.

Builds its own RAM uint8 cache at the requested size (../models/ab_res_cache_{S}.pt).
Run from src/:  python3 ab_resolution.py --sizes 224       # compares 224 vs the known 160=67.68
"""
import argparse, time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F

from ab_ordinal_test import (EffNet, EMA, DATA, BATCH, MAX_LR, WD, SEED, device, gpu_aug)

EPOCHS = 32
BASELINE_160 = 67.68   # Exp 33/35 SAFE softmax @S=160, same split


def build_cache(S):
    cf = Path(f'../models/ab_res_cache_{S}.pt')
    if cf.exists():
        b = torch.load(cf); print(f'  loaded {S} cache', flush=True); return b['X'], b['y']
    df = pd.read_csv(DATA / 'train.csv')
    X = torch.empty(len(df), 3, S, S, dtype=torch.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        a = np.asarray(Image.open(DATA / 'train' / row.id).convert('RGB').resize((S, S)))
        X[i] = torch.from_numpy(a.copy()).permute(2, 0, 1)
        if (i + 1) % 3000 == 0: print(f'    built {i+1}/{len(df)}', flush=True)
    y = torch.tensor(df['label'].values - 1)
    torch.save({'X': X, 'y': y}, cf)
    return X, y


def run_size(S, X, y, tr, va, mean, std):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = EffNet(5).to(device); ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    tr_t = torch.as_tensor(tr); steps = len(tr_t) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=EPOCHS * steps, pct_start=0.2)
    md = mean.to(device); sd = std.to(device)
    print(f'\n[S={S}] {sum(p.numel() for p in model.parameters())/1e6:.2f}M | {steps} batches/ep', flush=True)

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
            print(f'  [S={S}] ep{ep+1:2d}/{EPOCHS} exact {acc:5.2f} (best {best:5.2f})', flush=True)
    print(f'[S={S}] BEST exact {best:.2f}%  [{time.time()-t0:.0f}s]', flush=True)
    return best


def main():
    from sklearn.model_selection import StratifiedKFold
    ap = argparse.ArgumentParser()
    ap.add_argument('--sizes', default='224')  # comma list
    a = ap.parse_args()
    sizes = [int(s) for s in a.sizes.split(',')]
    print(f'device={device} EPOCHS={EPOCHS} sizes={sizes} (baseline S=160 = {BASELINE_160}%)', flush=True)
    res = {}
    for S in sizes:
        X, y = build_cache(S)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
        va = torch.as_tensor(va)
        px = X[torch.as_tensor(tr)].float()/255.
        mean = px.mean(dim=(0,2,3), keepdim=True)[0]; std = px.std(dim=(0,2,3), keepdim=True)[0]; del px
        res[S] = run_size(S, X, y, tr, va, mean, std)
    print('\n===== RESOLUTION A/B (exact-count acc) =====')
    print(f'  S=160 (ref)  {BASELINE_160:5.2f}%')
    for S, acc in res.items():
        print(f'  S={S:<8}  {acc:5.2f}%   (Δ vs 160 {acc-BASELINE_160:+.2f})')
    best_S = max(res, key=res.get)
    win = res[best_S] - BASELINE_160
    print(f'  VERDICT: {"resolution HELPS — bump Colab run resolution" if win >= 0.7 else "no clear resolution gain — keep 256"} '
          f'(best S={best_S} Δ{win:+.2f})')


if __name__ == '__main__':
    main()

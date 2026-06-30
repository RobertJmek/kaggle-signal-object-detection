"""
ASPECT-RATIO A/B (path-to-82) — does stretching the native 128x55 spectrogram to a SQUARE
distort the time/frequency geometry a counter depends on?

The deployment pipeline resizes native 128(freq)x55(time) -> square 256x256, stretching the
time axis ~4.6x and frequency ~2x (NON-uniform). For a COUNTING task, object separation along
time may matter, so this distortion could cost accuracy. Never isolated before (Exp 38 only
compared SQUARE sizes).

Clean isolation: MATCH the pixel budget (~16.4k px), vary ONLY the shape.
  SQUARE  128x128 = 16384 px   (distorted, time stretched 2.33x)
  NATIVE  196x84  = 16464 px   (H/W=2.33, matches native 128/55=2.327 -> no distortion)
Same compute, same EffNet B0-MBConv / softmax / count-preserving gpu_aug / EMA / OneCycle, same
StratifiedKFold(5, seed42) first split. The only variable is aspect.

Run from src/:  python3 ab_aspect.py
"""
import time
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from ab_ordinal_test import (EffNet, EMA, DATA, BATCH, MAX_LR, WD, SEED, device, gpu_aug)

EPOCHS = 32
# scale-up confirmation (~50k px, matched budget) — does the +3.55 @16k hold at deployment scale?
ARMS = {'SQUARE_224x224': (224, 224), 'NATIVE_342x147': (342, 147)}  # (H, W)


def build_cache(tag, H, W):
    cf = Path(f'../models/ab_aspect_{tag}.pt')
    if cf.exists():
        b = torch.load(cf); print(f'  loaded {tag} cache', flush=True); return b['X'], b['y']
    df = pd.read_csv(DATA / 'train.csv')
    X = torch.empty(len(df), 3, H, W, dtype=torch.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        a = np.asarray(Image.open(DATA / 'train' / row.id).convert('RGB').resize((W, H)))  # PIL=(W,H)
        X[i] = torch.from_numpy(a.copy()).permute(2, 0, 1)
        if (i + 1) % 3000 == 0: print(f'    built {i+1}/{len(df)}', flush=True)
    y = torch.tensor(df['label'].values - 1)
    torch.save({'X': X, 'y': y}, cf)
    return X, y


def run_arm(tag, H, W, X, y, tr, va, mean, std):
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = EffNet(5).to(device); ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    tr_t = torch.as_tensor(tr); steps = len(tr_t) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=EPOCHS * steps, pct_start=0.2)
    md = mean.to(device); sd = std.to(device)
    print(f'\n[{tag}] {H}x{W}={H*W}px | {sum(p.numel() for p in model.parameters())/1e6:.2f}M | {steps} batches/ep', flush=True)

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
    print(f'device={device} EPOCHS={EPOCHS} arms={list(ARMS)}', flush=True)
    res = {}
    for tag, (H, W) in ARMS.items():
        X, y = build_cache(tag, H, W)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
        va = torch.as_tensor(va)
        px = X[torch.as_tensor(tr)].float()/255.
        mean = px.mean(dim=(0,2,3), keepdim=True)[0]; std = px.std(dim=(0,2,3), keepdim=True)[0]; del px
        res[tag] = run_arm(tag, H, W, X, y, tr, va, mean, std)
    print('\n===== ASPECT-RATIO A/B (matched px budget, exact-count acc) =====')
    for tag, acc in res.items():
        print(f'  {tag:<16} {acc:5.2f}%', flush=True)
    sq = next(a for t, a in res.items() if t.startswith('SQUARE'))
    nat = next(a for t, a in res.items() if t.startswith('NATIVE'))
    d = nat - sq
    print(f'  Δ (NATIVE - SQUARE) = {d:+.2f}', flush=True)
    if d >= 0.7:
        print('  VERDICT: native aspect HELPS — switch deployment to native-aspect input (e.g. 256x110) & rerun.', flush=True)
    elif d <= -0.7:
        print('  VERDICT: square stretch HELPS (more time pixels > distortion cost) — keep square 256.', flush=True)
    else:
        print('  VERDICT: aspect is neutral (|Δ|<0.7) — distortion is not the lever; keep square 256.', flush=True)


if __name__ == '__main__':
    main()

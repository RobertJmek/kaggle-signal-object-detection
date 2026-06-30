"""
MULTI-RESOLUTION ENSEMBLE (path-to-80, follow-up to the failed arch-blend).

ab_ensemble_blend falsified ARCH diversity: a distinct net (DilatedCounter 69%) was weaker AND too
correlated (0.896) with EffNet, so blending hurt (-0.65). Better diversity axis = SCALE: blend the SAME
strong EffNet(w1.3) at two native resolutions (256x110 + 384x165). Both members are strong, and their
errors should decorrelate by receptive-field scale rather than by a capability gap.

Same StratifiedKFold(5,seed42) first split; index i is the same image in both caches (build_cache iterates
train.csv in order), so val softmax probs blend per-sample. Per-resolution mean/std.

Gate: blend >= best single + 0.5 -> add a 256-scale EffNet member to deployment (2x train cost, but both
members already are the deployed arch). Reuses train_capture from ab_ensemble_blend.

Run from src/:  python3 ab_ensemble_multires.py
"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from ab_aspect import build_cache
from ab_ensemble_blend import train_capture
from ab_ordinal_test import EffNet, SEED, device

ARMS = {'EffNet@256': (256, 110), 'EffNet@384': (384, 165)}


def prep(tag, H, W):
    X, y = build_cache(f'NAT_{H}x{W}', H, W)         # matches existing caches NAT_256x110 / NAT_384x165
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
    va = torch.as_tensor(va)
    px = X[torch.as_tensor(tr)].float()/255.
    mean = px.mean(dim=(0,2,3), keepdim=True)[0]; std = px.std(dim=(0,2,3), keepdim=True)[0]; del px
    return X, y, tr, va, mean.to(device), std.to(device)


def main():
    # w1.0 proxy: w1.3 @384 batch96 thrashed MPS memory (swapped). Scale-decorrelation is ~width-independent,
    # so w1.0 (4M, no swap) gives a valid go/no-go on whether multi-scale blending helps.
    print(f'multi-resolution ensemble: {list(ARMS)} (EffNet w1.0 proxy)', flush=True)
    res = {}
    for tag, (H, W) in ARMS.items():
        X, y, tr, va, md, sd = prep(tag, H, W)
        acc, probs = train_capture(tag, EffNet(5, width=1.0, depth_mult=1.0), X, y, tr, va, md, sd)
        res[tag] = (acc, probs, y[va])
        del X
    (a256, p256, yv) = res['EffNet@256']; (a384, p384, _) = res['EffNet@384']
    blend = 100*((p256 + p384).argmax(1) == yv).float().mean().item()
    corr = torch.corrcoef(torch.stack([p256.argmax(1).float(), p384.argmax(1).float()]))[0, 1].item()
    best_single = max(a256, a384)
    print('\n===== MULTI-RESOLUTION ENSEMBLE (exact-count acc) =====', flush=True)
    print(f'  EffNet@256   {a256:5.2f}%', flush=True)
    print(f'  EffNet@384   {a384:5.2f}%', flush=True)
    print(f'  blend        {blend:5.2f}%   (best single {best_single:.2f}, Δ {blend-best_single:+.2f})', flush=True)
    print(f'  pred correlation {corr:.3f}', flush=True)
    if blend >= best_single + 0.5:
        print('  VERDICT: multi-scale blend HELPS -> add a 256-scale EffNet member to deployment.', flush=True)
    else:
        print('  VERDICT: blend does not clear +0.5 -> single-scale 384 + folds + pseudo is the recipe.', flush=True)


if __name__ == '__main__':
    main()

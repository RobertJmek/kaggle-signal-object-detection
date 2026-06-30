"""
CAPACITY PUSH — where does width saturate? (follow-up to ab_capacity_native.py)

ab_capacity_native found width helps HARD at native aspect: w1.0=67.97% -> w1.3=71.42% (+3.45pp). Open
question: keep pushing or has it saturated? This trains a SINGLE new arm width=1.6/depth=1.2 (~16M) at the
same native 256x110 / aug / EMA / OneCycle / seed and compares against the KNOWN w1.3 reference (71.42%,
reused from the prior run — no need to retrain it).

Decision:
  WIDER - 71.42 >= +0.7  -> capacity still climbing; push deployment to w1.6 (or sweep higher).
  |Δ| < 0.7              -> saturating near w1.3; deploy w1.3 and spend budget on pseudo-R2 / ensemble.
  <= -0.7               -> w1.6 overfits; w1.3 is the sweet spot.

Run from src/:  python3 ab_capacity_push.py
"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from ab_aspect import build_cache
from ab_capacity_native import run_arm
from ab_ordinal_test import SEED

H, W = 256, 110
W13_REF = 71.42                                   # known width=1.3/depth=1.2 result (ab_capacity_native)
ARM = ('WIDER_w1.6', dict(width=1.6, depth_mult=1.2))


def main():
    tag, cfg = ARM
    print(f'capacity push @ native {H}x{W}: {tag} vs w1.3 ref {W13_REF:.2f}%', flush=True)
    X, y = build_cache('NAT_256x110', H, W)        # reuse cache
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
    va = torch.as_tensor(va)
    px = X[torch.as_tensor(tr)].float()/255.
    mean = px.mean(dim=(0,2,3), keepdim=True)[0]; std = px.std(dim=(0,2,3), keepdim=True)[0]; del px
    acc = run_arm(tag, cfg, X, y, tr, va, mean, std)
    d = acc - W13_REF
    print('\n===== CAPACITY PUSH (exact-count acc) =====', flush=True)
    print(f'  w1.3 (ref)   {W13_REF:5.2f}%', flush=True)
    print(f'  {tag:<12} {acc:5.2f}%   Δ vs w1.3 = {d:+.2f}', flush=True)
    if d >= 0.7:
        print('  VERDICT: still climbing -> push deployment to w1.6 (and consider sweeping higher).', flush=True)
    elif d <= -0.7:
        print('  VERDICT: w1.6 overfits -> w1.3 is the sweet spot; deploy w1.3.', flush=True)
    else:
        print('  VERDICT: saturating near w1.3 -> deploy w1.3; spend budget on pseudo-R2 / ensemble.', flush=True)


if __name__ == '__main__':
    main()

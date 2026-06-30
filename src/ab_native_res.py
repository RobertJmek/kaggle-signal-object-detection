"""
NATIVE-ASPECT RESOLUTION sweep (path-to-82, follow-up to Exp 41).

Aspect ratio is a confirmed lever (README Exp 41: native beats square +1.77pp @50k). But the deployment
resolution 384x165 was chosen only to MATCH the old 256^2 pixel budget — never measured. And Exp 38's
"256 is optimal" was on DISTORTED square images, so that saturation point is suspect at native aspect.

This sweeps resolution AT FIXED native aspect (H/W = 2.327, matching native 128x55) to find the real
optimum: does less upscaling (128x56, near-native, sharper thin objects, cheaper) match or beat the
heavier 384x165? Same EffNet/aug/EMA/OneCycle, same StratifiedKFold(5, seed42) first split.

Reuses the ab_aspect.py machinery. Run from src/:  python3 ab_native_res.py
"""
import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from ab_aspect import build_cache, run_arm
from ab_ordinal_test import SEED

# all H/W ~= 2.327 (native). spans near-native -> current deployment.
SIZES = {'NAT_128x56': (128, 56), 'NAT_256x110': (256, 110), 'NAT_384x165': (384, 165)}


def main():
    print(f'native-aspect resolution sweep: {list(SIZES)}', flush=True)
    res = {}
    for tag, (H, W) in SIZES.items():
        X, y = build_cache(tag, H, W)
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
        tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
        va = torch.as_tensor(va)
        px = X[torch.as_tensor(tr)].float() / 255.
        mean = px.mean(dim=(0, 2, 3), keepdim=True)[0]; std = px.std(dim=(0, 2, 3), keepdim=True)[0]; del px
        res[tag] = run_arm(tag, H, W, X, y, tr, va, mean, std)
    print('\n===== NATIVE-ASPECT RESOLUTION SWEEP (exact-count acc) =====')
    for tag, acc in res.items():
        px = SIZES[tag][0] * SIZES[tag][1]
        print(f'  {tag:<14} ({px//1000}k px)  {acc:5.2f}%', flush=True)
    best = max(res, key=res.get)
    cur = res['NAT_384x165']
    print(f'  best = {best} ({res[best]:.2f}%); current deployment NAT_384x165 = {cur:.2f}%', flush=True)
    if best != 'NAT_384x165' and res[best] >= cur - 0.3:
        print(f'  VERDICT: {best} matches/beats 384x165 -> switch (cheaper/sharper). Δ {res[best]-cur:+.2f}', flush=True)
    elif res['NAT_384x165'] == max(res.values()):
        print('  VERDICT: 384x165 is the optimum among these -> keep it.', flush=True)
    else:
        print('  VERDICT: see table; pick best acc vs compute tradeoff.', flush=True)


if __name__ == '__main__':
    main()

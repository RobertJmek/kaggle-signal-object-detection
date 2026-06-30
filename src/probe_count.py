"""
Phase 0 counting probe (README Exp 32 / path-to-80 campaign).

Hypothesis: the label is a COUNT of signal objects (thin oriented lines). Exp 29's naive
connected-component count had corr -0.04 with the label — but that thresholded raw pixels and
counted noise blobs. Here we use an oriented Gabor matched-filter bank (the same design as the
proposed OrientedStem front-end) to suppress noise and respond to thin lines, then extract simple
counting/energy features and ask: does a shallow model (RandomForest) recover the count?

Gate: if RF on these hand features beats chance (20%) clearly (>~30-35%) and per-orientation energy
correlates with the label, the count hypothesis + oriented-front-end direction are confirmed.

Run from src/:  python3 probe_count.py
Read-only on the data; writes nothing except prints.
"""
import math, os
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn.functional as F

DATA = '../data/raw'
N_SAMPLE = 3000           # train images to probe
SEED = 42


def build_gabor(n_orient=12, scales=(2.0, 3.0, 5.0), k=9):
    """Zero-DC oriented line/ridge filters -> [n,1,k,k] torch tensor (same recipe as OrientedStem)."""
    half = k // 2
    ys, xs = torch.meshgrid(torch.arange(-half, half + 1).float(),
                            torch.arange(-half, half + 1).float(), indexing='ij')
    filters = []
    for lam in scales:
        sigma = 0.56 * lam
        gamma = 0.5
        for o in range(n_orient):
            theta = math.pi * o / n_orient
            xr = xs * math.cos(theta) + ys * math.sin(theta)
            yr = -xs * math.sin(theta) + ys * math.cos(theta)
            env = torch.exp(-(xr ** 2 + (gamma * yr) ** 2) / (2 * sigma ** 2))
            carrier = torch.cos(2 * math.pi * xr / lam)
            g = env * carrier
            g = g - g.mean()
            g = g / (g.norm() + 1e-6)
            filters.append(g)
    return torch.stack(filters).unsqueeze(1), n_orient, len(scales)


def load_g(fid):
    a = np.array(Image.open(os.path.join(DATA, 'train', fid)))[:, :, 1].astype(np.float32) / 255.0
    return a  # 128x55


def features(img_t, bank, n_orient, n_scale):
    """img_t: [1,1,H,W]. Return a feature vector summarizing oriented responses."""
    with torch.no_grad():
        r = F.conv2d(img_t, bank, padding=bank.shape[-1] // 2).abs()  # [1,n,H,W]
        n = r.shape[1]
        # per-filter energy + peak stats
        e_mean = r.mean(dim=(2, 3)).flatten()                 # n
        e_max = r.amax(dim=(2, 3)).flatten()                  # n
        # max-response map across all filters
        mx = r.amax(dim=1, keepdim=True)                      # [1,1,H,W]
        # peak count at several thresholds via local-max + threshold
        pooled = F.max_pool2d(mx, 3, stride=1, padding=1)
        is_peak = (mx == pooled).float()
        peaks = []
        for q in (0.90, 0.95, 0.975, 0.99):
            thr = torch.quantile(mx.flatten(), q)
            peaks.append((is_peak * (mx > thr)).sum().item())
        # per-orientation aggregated energy (sum over scales) — orientation profile
        ori = r.view(1, n_scale, n_orient, *r.shape[2:]).mean(dim=(1, 3, 4)).flatten()  # n_orient
        feat = torch.cat([e_mean, e_max, ori, torch.tensor(peaks)]).numpy()
    return feat


def main():
    df = pd.read_csv(os.path.join(DATA, 'train.csv'))
    df = df.sample(N_SAMPLE, random_state=SEED).reset_index(drop=True)
    bank, n_orient, n_scale = build_gabor()
    print(f'Gabor bank: {bank.shape[0]} filters ({n_orient} orient x {n_scale} scale), k={bank.shape[-1]}')

    X = np.zeros((len(df), bank.shape[0] * 2 + n_orient + 4), np.float32)
    y = df['label'].values
    for i, fid in enumerate(df['id']):
        g = torch.from_numpy(load_g(fid))[None, None]
        X[i] = features(g, bank, n_orient, n_scale)
        if (i + 1) % 1000 == 0:
            print(f'  {i+1}/{len(df)}')

    # correlation of total oriented energy with label
    total_energy = X[:, :bank.shape[0]].sum(1)
    peak_feat = X[:, -1]  # peaks at q=0.99
    from scipy.stats import spearmanr
    print(f'\nspearman(total oriented energy, label) = {spearmanr(total_energy, y).correlation:+.3f}')
    print(f'spearman(peak-count@0.99,      label) = {spearmanr(peak_feat, y).correlation:+.3f}')

    # RandomForest 5-class accuracy (5-fold CV) on the hand features
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    rf = RandomForestClassifier(n_estimators=300, max_depth=12, n_jobs=-1, random_state=SEED)
    acc = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f'\nRandomForest 5-class acc (5-fold CV) on Gabor features: '
          f'{acc.mean()*100:.2f}% +/- {acc.std()*100:.2f}  (chance=20%, Exp29 naive=~23%)')
    print('\nGATE: >~30-35% confirms the count signal is extractable from oriented responses')
    print('      -> validates the ordinal-head + oriented-front-end direction.')


if __name__ == '__main__':
    main()

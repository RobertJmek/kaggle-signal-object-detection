"""
Phase 0b probe: count the straight-line OBJECTS directly via a Radon line detector.

Rationale: the label is a COUNT of signal objects, and the objects are straight lines (vertical/
horizontal carriers + diagonal chirps; a linear chirp is a straight line in a spectrogram). Exp 29
tried GLOBAL FFT-angular energy (failed) but never counted individual line peaks. Here we:
  1. enhance lines + suppress the variable noise floor with a Gabor max-response map,
  2. binarize the enhanced map,
  3. Radon-transform it (project along many angles) -> sinogram,
  4. each strong, well-separated peak in (angle, offset) space = one line object; count them,
  5. ask whether that count tracks the label (corr + RF accuracy).

If line-count correlates strongly, a detection/density-counting architecture is the route to ~80.
Run from src/:  python3 probe_hough.py
"""
import math
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.ndimage import rotate, maximum_filter, gaussian_filter
import torch
import torch.nn.functional as F

from probe_count import build_gabor, load_g
DATA = Path('../data/raw')

N_SAMPLE = 1500
SEED = 42
N_ANGLES = 72
np.random.seed(SEED)


def gabor_enhance(g, bank):
    """g: HxW float -> line-enhanced map (max abs oriented response), noise-suppressed."""
    t = torch.from_numpy(g)[None, None]
    with torch.no_grad():
        r = F.conv2d(t, bank, padding=bank.shape[-1] // 2).abs().amax(1)[0].numpy()
    return r


def radon(mask, n_angles=N_ANGLES):
    """Simple Radon: for each angle, rotate and sum along rows -> sinogram [n_angles, W']."""
    angles = np.linspace(0., 180., n_angles, endpoint=False)
    proj = []
    for a in angles:
        rot = rotate(mask, a, reshape=True, order=1, mode='constant', cval=0.0)
        proj.append(rot.sum(0))
    W = max(p.shape[0] for p in proj)
    sino = np.zeros((n_angles, W), np.float32)
    for i, p in enumerate(proj):
        off = (W - p.shape[0]) // 2
        sino[i, off:off + p.shape[0]] = p
    return sino, angles


def count_lines(sino, rel_thr=0.45, min_sep=4):
    """Count well-separated peaks in the sinogram above rel_thr * global max."""
    s = gaussian_filter(sino, 1.0)
    thr = rel_thr * s.max()
    mx = maximum_filter(s, size=(3, min_sep))
    peaks = (s == mx) & (s > thr)
    # collapse near-duplicate peaks along the offset axis per angle row
    return int(peaks.sum())


def features(g, bank):
    enh = gabor_enhance(g, bank)
    enh = enh / (enh.max() + 1e-6)
    feats = []
    counts = []
    for q in (0.85, 0.92, 0.96):
        mask = (enh > q).astype(np.float32)
        sino, _ = radon(mask)
        for rt in (0.40, 0.55):
            counts.append(count_lines(sino, rel_thr=rt))
        feats.append(sino.max())
        feats.append(mask.sum())
    return np.array(counts + feats, np.float32), counts


def main():
    df = pd.read_csv(DATA / 'train.csv').sample(N_SAMPLE, random_state=SEED).reset_index(drop=True)
    bank, _, _ = build_gabor(n_orient=8, scales=(2.0, 4.0), k=9)
    y = df['label'].values
    X = []
    primary = []   # one representative line-count (q=0.92, rel_thr 0.40)
    for i, fid in enumerate(df['id']):
        g = load_g(fid)
        f, counts = features(g, bank)
        X.append(f); primary.append(counts[2])
        if (i + 1) % 300 == 0:
            print(f'  {i+1}/{len(df)}', flush=True)
    X = np.stack(X); primary = np.array(primary)

    from scipy.stats import spearmanr
    print(f'\nspearman(primary line-count, label) = {spearmanr(primary, y).correlation:+.3f}')
    # mean predicted line-count per true label (should increase 1->5 if it works)
    print('mean line-count by true label:')
    for c in [1, 2, 3, 4, 5]:
        print(f'  label {c}: mean count = {primary[y==c].mean():.2f}')

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    rf = RandomForestClassifier(n_estimators=300, max_depth=10, n_jobs=-1, random_state=SEED)
    acc = cross_val_score(rf, X, y, cv=5, scoring='accuracy', n_jobs=-1)
    print(f'\nRandomForest 5-class acc on Radon line-count features: {acc.mean()*100:.2f}% +/- {acc.std()*100:.2f}')
    print('  (chance 20%, Exp29 naive 23%, Gabor-energy RF 29.6%)')
    print('GATE: spearman > ~0.4 OR RF > ~40% => detection/density-counting is the route to ~80.')


if __name__ == '__main__':
    main()

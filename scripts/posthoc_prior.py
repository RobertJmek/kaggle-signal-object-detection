"""
Post-hoc PRIOR CORRECTION for Run 27 (path-to-80) — fix the label-1 over-prediction bias.

The trained 3-fold ensemble predicts label 1 at ~35% of test, but the train set is ~uniform
(3500/3000/3000/3000/3000 = 22.6/19.4/19.4/19.4/19.4%). Test is almost certainly the same shape, so
the model has a calibration bias toward count=1. Fix WITHOUT retraining: reweight class probabilities
so the predicted marginal matches the known prior, then re-argmax (Sinkhorn / iterative proportional
fitting). The class weights depend ONLY on the predicted probs + the known target prior — NOT on any
labels — so validating on the out-of-fold val sets is leakage-free.

Procedure:
  1. Rebuild the 3 folds, load best_R0f{0,1,2}.pt, predict each fold's OWN val set -> OOF val probs.
  2. Baseline OOF val acc vs PRIOR-CORRECTED OOF val acc (target = train prior). Report delta.
  3. If it helps, recompute test probs (ensemble + translation TTA), correct, and write
     submission_corrected.csv alongside the original.

Run from the box (same env as run27):
  DATA_PATH=/root/comp/data OUT_DIR=/root/run27_out python posthoc_prior.py
"""
import os, glob, math
from pathlib import Path
import numpy as np, pandas as pd
import torch, torch.nn as nn
from PIL import Image
from sklearn.model_selection import StratifiedKFold

OUT_DIR = Path(os.environ.get('OUT_DIR', './run27_out'))
DATA_DIR = Path(os.environ['DATA_PATH'])
IMG = 256
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = (device.type == 'cuda')
# CONFIG must match run27_vast.py
DROPOUT, DROP_PATH, WIDTH, DEPTH = 0.4, 0.2, 1.1, 1.2

# ----- model (copied verbatim from run27_vast.py so we can load the saved state dicts) -----
class DropPath(nn.Module):
    def __init__(s, p=0.0): super().__init__(); s.p = p
    def forward(s, x): return x
class MBConv(nn.Module):
    def __init__(s, cin, cout, k=3, st=1, t=6, se=0.25, dp=0.0):
        super().__init__()
        cm = cin * t; s.use_res = (st == 1 and cin == cout); L = []
        if t != 1: L += [nn.Conv2d(cin, cm, 1, bias=False), nn.BatchNorm2d(cm), nn.SiLU(True)]
        L += [nn.Conv2d(cm, cm, k, st, k // 2, groups=cm, bias=False), nn.BatchNorm2d(cm), nn.SiLU(True)]
        s.conv = nn.Sequential(*L); sc = max(1, int(cin * se))
        s.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cm, sc, 1), nn.SiLU(True), nn.Conv2d(sc, cm, 1), nn.Sigmoid())
        s.proj = nn.Sequential(nn.Conv2d(cm, cout, 1, bias=False), nn.BatchNorm2d(cout)); s.dp = DropPath(dp)
    def forward(s, x):
        o = s.conv(x); o = o * s.se(o); o = s.proj(o); return x + o if s.use_res else o
class EffNet(nn.Module):
    cfg = [(1, 16, 3, 1, 1), (6, 24, 3, 2, 2), (6, 40, 5, 2, 2), (6, 80, 3, 2, 3),
           (6, 112, 5, 1, 3), (6, 192, 5, 2, 4), (6, 320, 3, 1, 1)]
    def __init__(s, num_classes=5, dropout=0.3, drop_path=0.1, width=1.0, depth_mult=1.0):
        super().__init__()
        ch = lambda c: int(c * width); rep = lambda r: int(math.ceil(r * depth_mult))
        s.stem = nn.Sequential(nn.Conv2d(3, ch(32), 3, 2, 1, bias=False), nn.BatchNorm2d(ch(32)), nn.SiLU(True))
        blocks = []; cin = ch(32); tot = sum(rep(r) for *_, r in s.cfg); bi = 0
        for t, co, k, st, r in s.cfg:
            co = ch(co)
            for j in range(rep(r)):
                blocks.append(MBConv(cin, co, k, st if j == 0 else 1, t, dp=drop_path * bi / max(1, tot - 1))); cin = co; bi += 1
        s.blocks = nn.Sequential(*blocks)
        s.head = nn.Sequential(nn.Conv2d(cin, ch(1280), 1, bias=False), nn.BatchNorm2d(ch(1280)), nn.SiLU(True),
                               nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(ch(1280), num_classes))
    def forward(s, x): return s.head(s.blocks(s.stem(x)))

# ----- data: resolve dir, read csvs, RGB stats, cache uint8 -----
hits = sorted(glob.glob(str(DATA_DIR) + '/**/train.csv', recursive=True), key=len)
DATA_DIR = Path(hits[0]).parent
def imgdir(stem):
    d = DATA_DIR / stem
    if d.is_dir(): return d
    return [Path(x) for x in glob.glob(str(DATA_DIR) + '/**/' + stem, recursive=True) if Path(x).is_dir()][0]
TRAIN_DIR, TEST_DIR = imgdir('train'), imgdir('test')
train_df = pd.read_csv(DATA_DIR / 'train.csv'); test_df = pd.read_csv(DATA_DIR / 'test.csv')
_smp = train_df['id'].sample(800, random_state=0); _a = np.zeros(3); _a2 = np.zeros(3); _n = 0
for fn in _smp:
    im = np.asarray(Image.open(TRAIN_DIR / fn).convert('RGB').resize((IMG, IMG)), np.float32) / 255.
    _a += im.reshape(-1, 3).sum(0); _a2 += (im.reshape(-1, 3) ** 2).sum(0); _n += IMG * IMG
MEAN = torch.tensor(_a / _n, dtype=torch.float32).view(3, 1, 1).to(device)
STD = torch.tensor(np.sqrt(_a2 / _n - (_a / _n) ** 2), dtype=torch.float32).view(3, 1, 1).to(device)
def cache_u8(df, d):
    X = torch.empty(len(df), 3, IMG, IMG, dtype=torch.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        X[i] = torch.from_numpy(np.asarray(Image.open(d / row.id).convert('RGB').resize((IMG, IMG))).copy()).permute(2, 0, 1)
    return X
print('caching...', flush=True)
XTR = cache_u8(train_df, TRAIN_DIR); XTE = cache_u8(test_df, TEST_DIR)
YTR = torch.tensor(train_df['label'].values - 1)
TARGET = torch.tensor(np.bincount(YTR.numpy(), minlength=5) / len(YTR), dtype=torch.float32)  # train prior
print('train prior (target):', [f'{p:.3f}' for p in TARGET.tolist()], flush=True)

def load_fold(tag):
    m = EffNet(5, DROPOUT, DROP_PATH, width=WIDTH, depth_mult=DEPTH).to(device)
    m.load_state_dict(torch.load(OUT_DIR / f'best_{tag}.pt', map_location=device)); m.eval(); return m

@torch.no_grad()
def probs_of(m, X, idx, tta=False):
    shifts = [(0, 0), (0, 12), (0, -12), (6, 0), (-6, 0)] if tta else [(0, 0)]
    out = torch.zeros(len(idx), 5)
    for s0 in range(0, len(idx), 256):
        bi = idx[s0:s0 + 256]
        x0 = (X[bi].to(device).float() / 255. - MEAN) / STD
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            for fs, ts in shifts:
                x = torch.roll(x0, shifts=(fs, ts), dims=(2, 3))
                out[s0:s0 + 256] += torch.softmax(m(x).float(), 1).cpu()
    return out / len(shifts)

def fit_weights(probs, target, iters=300):
    """iterative proportional fitting: find per-class w so (probs*w) marginal == target."""
    w = torch.ones(probs.shape[1])
    for _ in range(iters):
        adj = probs * w; adj = adj / adj.sum(1, keepdim=True)
        cur = adj.mean(0)
        w = w * (target / (cur + 1e-12)); w = w / w.mean()
    return w
def apply_w(probs, w):
    adj = probs * w; return adj / adj.sum(1, keepdim=True)

# ----- 1. OOF val: baseline vs corrected (leakage-free: weights use only probs + known prior) -----
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
splits = list(skf.split(np.arange(len(train_df)), train_df['label']))
oof_p = torch.zeros(len(train_df), 5); oof_y = YTR.clone()
for f, (tri, vai) in enumerate(splits):
    m = load_fold(f'R0f{f}')
    oof_p[torch.as_tensor(vai)] = probs_of(m, XTR, torch.as_tensor(vai), tta=False)
base_acc = 100 * (oof_p.argmax(1) == oof_y).float().mean().item()
w = fit_weights(oof_p, TARGET)
corr_acc = 100 * (apply_w(oof_p, w).argmax(1) == oof_y).float().mean().item()
print(f'\nOOF val  baseline {base_acc:.2f}%  ->  prior-corrected {corr_acc:.2f}%   (Δ {corr_acc-base_acc:+.2f})', flush=True)
print(f'class weights: {[round(x,3) for x in w.tolist()]}', flush=True)
print(f'raw val pred dist : {[round(x,3) for x in oof_p.argmax(1).bincount(minlength=5).div(len(oof_y)).tolist()]}', flush=True)
print(f'corr val pred dist: {[round(x,3) for x in apply_w(oof_p,w).argmax(1).bincount(minlength=5).div(len(oof_y)).tolist()]}', flush=True)

# ----- 2. apply to test (ensemble + TTA), write corrected submission -----
test_p = torch.zeros(len(test_df), 5)
for f in range(3):
    test_p += probs_of(load_fold(f'R0f{f}'), XTE, torch.arange(len(test_df)), tta=True)
test_p /= 3
wt = fit_weights(test_p, TARGET)
pred_raw = test_p.argmax(1).numpy() + 1
pred_corr = apply_w(test_p, wt).argmax(1).numpy() + 1
pd.DataFrame({'id': test_df['id'].tolist(), 'label': pred_corr}).to_csv(OUT_DIR / 'submission_corrected.csv', index=False)
print(f'\nraw  test dist: {np.bincount(pred_raw, minlength=6)[1:].tolist()}', flush=True)
print(f'corr test dist: {np.bincount(pred_corr, minlength=6)[1:].tolist()}', flush=True)
print(f'changed {int((pred_raw!=pred_corr).sum())}/{len(pred_corr)} test predictions', flush=True)
print(f'-> wrote {OUT_DIR}/submission_corrected.csv', flush=True)
print('\nDECISION:', 'SUBMIT submission_corrected.csv (val improved)' if corr_acc > base_acc + 0.1
      else 'keep original submission.csv (no val gain — bias may be test-only; could still try on Kaggle)', flush=True)

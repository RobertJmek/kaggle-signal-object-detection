"""
CORRECTED Run 30 submission. The TTA probe (tta_probe.py) proved multi-scale {384,448,512} HURTS
(-1.88pp val) while translation TTA HELPS (+0.44pp). Run 30's own submission.csv used the full
multi-scale config (-1.37pp below plain 448) and must NOT be uploaded.

This regenerates submission.csv with the CORRECT recipe: single-scale 448 + translation TTA only,
ensembling the R1 fold models. Same preprocessing as run30_vast.py / tta_probe.py.

    DATA_PATH=./data/raw OUT_DIR=./models/run30_out python3 make_submission_corrected.py
    # optional: ROUND=R0  (default R1 if present)  |  OUT_SUB=./submission_run30_corrected.csv
"""
import os, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

OUT_DIR = Path(os.environ.get('OUT_DIR', './models/run30_out'))
OUT_SUB = Path(os.environ.get('OUT_SUB', './submission_run30_corrected.csv'))
IMG_H, IMG_W = 448, 192
TRAIN_FOLDS = int(os.environ.get('TRAIN_FOLDS', '2'))
WIDTH, DEPTH, DROPOUT, DROP_PATH = 1.3, 1.2, 0.4, 0.2
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available() else 'cpu')
use_amp = (device.type == 'cuda')
print('torch', torch.__version__, '| device:', device, '| OUT_DIR:', OUT_DIR, flush=True)

# ----------------------------- data (identical preprocessing) -----------------------------
_dp = os.environ.get('DATA_PATH')
if _dp:
    DATA_DIR = Path(_dp)
else:
    import kagglehub
    DATA_DIR = Path(kagglehub.competition_download('signal-object-detection'))
hits = sorted(glob.glob(str(DATA_DIR) + '/**/train.csv', recursive=True), key=len)
assert hits, f'train.csv not found under {DATA_DIR}'
DATA_DIR = Path(hits[0]).parent
def imgdir(stem):
    d = DATA_DIR / stem
    if d.is_dir(): return d
    cands = [Path(x) for x in glob.glob(str(DATA_DIR) + '/**/' + stem, recursive=True) if Path(x).is_dir()]
    return cands[0] if cands else d
TRAIN_DIR, TEST_DIR = imgdir('train'), imgdir('test')
train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df  = pd.read_csv(DATA_DIR / 'test.csv')

# seed-0 800-sample RGB mean/std — identical to run30 (deterministic)
_smp = train_df['id'].sample(800, random_state=0); _acc = np.zeros(3); _acc2 = np.zeros(3); _n = 0
for fn in _smp:
    a = np.asarray(Image.open(TRAIN_DIR / fn).convert('RGB').resize((IMG_W, IMG_H)), np.float32) / 255.
    _acc += a.reshape(-1, 3).sum(0); _acc2 += (a.reshape(-1, 3) ** 2).sum(0); _n += IMG_H * IMG_W
MEAN_d = torch.tensor(_acc / _n, dtype=torch.float32).view(3, 1, 1).to(device)
STD_d  = torch.tensor(np.sqrt(_acc2 / _n - (_acc / _n) ** 2), dtype=torch.float32).view(3, 1, 1).to(device)

def cache_u8(df, img_dir):
    X = torch.empty(len(df), 3, IMG_H, IMG_W, dtype=torch.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        a = np.asarray(Image.open(img_dir / row.id).convert('RGB').resize((IMG_W, IMG_H)))
        X[i] = torch.from_numpy(a.copy()).permute(2, 0, 1)
    return X
print('caching test...', flush=True); XTE = cache_u8(test_df, TEST_DIR)

# ----------------------------- model (identical to run30_vast.py) -----------------------------
class DropPath(nn.Module):
    def __init__(s, p=0.0): super().__init__(); s.p = p
    def forward(s, x):
        if s.p == 0.0 or not s.training: return x
        k = 1 - s.p; m = torch.empty((x.size(0), 1, 1, 1), dtype=x.dtype, device=x.device).bernoulli_(k); return x / k * m

class MBConv(nn.Module):
    def __init__(s, cin, cout, k=3, st=1, t=6, se=0.25, dp=0.0):
        super().__init__()
        cm = cin * t; s.use_res = (st == 1 and cin == cout); L = []
        if t != 1: L += [nn.Conv2d(cin, cm, 1, bias=False), nn.BatchNorm2d(cm), nn.SiLU(True)]
        L += [nn.Conv2d(cm, cm, k, st, k // 2, groups=cm, bias=False), nn.BatchNorm2d(cm), nn.SiLU(True)]
        s.conv = nn.Sequential(*L); sc = max(1, int(cin * se))
        s.se = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(cm, sc, 1), nn.SiLU(True), nn.Conv2d(sc, cm, 1), nn.Sigmoid())
        s.proj = nn.Sequential(nn.Conv2d(cm, cout, 1, bias=False), nn.BatchNorm2d(cout)); s.dp = DropPath(dp)
        if s.use_res: nn.init.zeros_(s.proj[1].weight)
    def forward(s, x):
        o = s.conv(x); o = o * s.se(o); o = s.proj(o); return x + s.dp(o) if s.use_res else o

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

# ----------------------------- which round -----------------------------
def round_complete(tag):
    return all((OUT_DIR / f'best_{tag}f{f}.pt').exists() for f in range(TRAIN_FOLDS))
ROUND = os.environ.get('ROUND')
if not ROUND:
    for cand in ['R3', 'R2', 'R1', 'R0']:
        if round_complete(cand): ROUND = cand; break
assert ROUND and round_complete(ROUND), f'no complete fold set in {OUT_DIR}'
fold_states = [torch.load(OUT_DIR / f'best_{ROUND}f{f}.pt', map_location=device) for f in range(TRAIN_FOLDS)]
print(f'ensembling {len(fold_states)} folds from round {ROUND}', flush=True)

# ----------------------------- inference: 448 + translation TTA ONLY (NO multi-scale) -----------------------------
TTA = [(0, 0), (0, 12), (0, -12), (6, 0), (-6, 0)]   # translation TTA (probe: +0.44). multi-scale REMOVED (probe: -1.88).
probs = torch.zeros(len(test_df), 5)
for st in fold_states:
    m = EffNet(5, DROPOUT, DROP_PATH, width=WIDTH, depth_mult=DEPTH).to(device)
    m.load_state_dict(st); m.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for s0 in range(0, len(test_df), 128):
            xs = (XTE[s0:s0 + 128].to(device).float() / 255. - MEAN_d) / STD_d   # native 448x192, no rescale
            for fs, ts in TTA:
                x = torch.roll(xs, shifts=(fs, ts), dims=(2, 3))
                probs[s0:s0 + 128] += torch.softmax(m(x).float(), 1).cpu()
print(f'inference: {len(fold_states)} folds x 1 scale (448) x {len(TTA)} TTA = {len(fold_states)*len(TTA)} passes/image', flush=True)
pred = probs.argmax(1).numpy() + 1
sub = pd.DataFrame({'id': test_df['id'].tolist(), 'label': pred})
sub.to_csv(OUT_SUB, index=False)
print(sub['label'].value_counts().sort_index(), flush=True)
print('-> wrote', OUT_SUB, '(UPLOAD THIS, not run30_out/submission.csv)', flush=True)

"""
TTA PROBE (zero-submission). Measures the multi-scale + translation TTA gain on the HELD-OUT VAL folds,
so we know what Run 30's submission-time TTA actually buys before spending a Kaggle slot.

Why this is leakage-free: each Run 30 fold model was trained on splits[f]=(tri,vai); we evaluate fold f's
saved model ONLY on its own vai. (Pseudo-round R1 models also trained on the same splits' real data plus
test-only pseudo, which is disjoint from vai — so vai stays clean for R1 too.)

It reproduces Run 30's preprocessing EXACTLY (IMG 448x192, seed-0 800-sample RGB mean/std, the same
StratifiedKFold(5, seed42) first TRAIN_FOLDS splits, same EffNet w1.3/d1.2). The 'base' config
(single 448, no TTA) should reproduce each fold's printed `[tag] BEST` within ~0.1pp — a sanity anchor,
since best_*.pt is exactly the best-val checkpoint evaluate_cache selected.

Four configs, each fold evaluated on its own val, then averaged across folds (= the CV metric):
  base        single-scale 448, no TTA          (1 pass)   -> anchor, ~= printed CV
  +trans      448 + translation TTA             (5 passes) -> translation contribution
  +multiscale {384,448,512}, no translation     (3 passes) -> multi-scale contribution
  full        {384,448,512} x translation       (15 passes)-> exactly what submission.csv does

Run on the GPU box, AFTER Run 30 has written best_*.pt:
    DATA_PATH=/root/comp/data OUT_DIR=./run30_out python tta_probe.py 2>&1 | tee tta_probe.log
    # optional: ROUND=R1   (default: auto-detect highest complete round on disk)
"""
import os, glob, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold

OUT_DIR = Path(os.environ.get('OUT_DIR', './run30_out'))
IMG_H, IMG_W = 448, 192
TRAIN_FOLDS = int(os.environ.get('TRAIN_FOLDS', '2'))
N_SPLITS = 5
# must match run30 CONFIG for load_state_dict to line up
WIDTH, DEPTH, DROPOUT, DROP_PATH = 1.3, 1.2, 0.4, 0.2
device = torch.device('cuda' if torch.cuda.is_available()
                      else 'mps' if torch.backends.mps.is_available() else 'cpu')
use_amp = (device.type == 'cuda')   # fp16 autocast on CUDA only; MPS/CPU run fp32
print('torch', torch.__version__, '| device:', device, '| OUT_DIR:', OUT_DIR, flush=True)

# ----------------------------- data (identical to run30_vast.py) -----------------------------
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
TRAIN_DIR = imgdir('train')
train_df = pd.read_csv(DATA_DIR / 'train.csv')

# seed-0 800-sample RGB mean/std — identical formula to run30 (deterministic -> same values)
_smp = train_df['id'].sample(800, random_state=0); _acc = np.zeros(3); _acc2 = np.zeros(3); _n = 0
for fn in _smp:
    a = np.asarray(Image.open(TRAIN_DIR / fn).convert('RGB').resize((IMG_W, IMG_H)), np.float32) / 255.
    _acc += a.reshape(-1, 3).sum(0); _acc2 += (a.reshape(-1, 3) ** 2).sum(0); _n += IMG_H * IMG_W
MEAN = torch.tensor(_acc / _n, dtype=torch.float32).view(3, 1, 1)
STD  = torch.tensor(np.sqrt(_acc2 / _n - (_acc / _n) ** 2), dtype=torch.float32).view(3, 1, 1)
MEAN_d = MEAN.to(device); STD_d = STD.to(device)
print('RGB mean', MEAN.flatten().tolist(), 'std', STD.flatten().tolist(), flush=True)

def cache_u8(df, img_dir):
    X = torch.empty(len(df), 3, IMG_H, IMG_W, dtype=torch.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        a = np.asarray(Image.open(img_dir / row.id).convert('RGB').resize((IMG_W, IMG_H)))
        X[i] = torch.from_numpy(a.copy()).permute(2, 0, 1)
    return X
print('caching train...', flush=True); XTR = cache_u8(train_df, TRAIN_DIR)
YTR = torch.tensor(train_df['label'].values - 1)

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

# ----------------------------- splits (identical to run30_vast.py) -----------------------------
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(skf.split(np.arange(len(train_df)), train_df['label']))[:TRAIN_FOLDS]

# ----------------------------- which round's models to probe -----------------------------
def round_complete(tag):
    return all((OUT_DIR / f'best_{tag}f{f}.pt').exists() for f in range(TRAIN_FOLDS))
ROUND = os.environ.get('ROUND')
if not ROUND:
    for cand in ['R3', 'R2', 'R1', 'R0']:        # prefer highest complete round (likely the promoted/submitted one)
        if round_complete(cand): ROUND = cand; break
assert ROUND and round_complete(ROUND), f'no complete fold set found in {OUT_DIR} (looked for best_R*f*.pt)'
print(f'probing round {ROUND}  (set ROUND=R0 to compare; match this to the PROMOTED line in run30.log)', flush=True)

# ----------------------------- TTA configs -----------------------------
FULL_SCALES = [(384, 165), (448, 192), (512, 220)]
ONE_SCALE   = [(448, 192)]
FULL_TTA    = [(0, 0), (0, 12), (0, -12), (6, 0), (-6, 0)]
NO_TTA      = [(0, 0)]
CONFIGS = {
    'base        (448, no TTA)        ': (ONE_SCALE,  NO_TTA),
    '+trans      (448, transTTA)      ': (ONE_SCALE,  FULL_TTA),
    '+multiscale ({384,448,512})      ': (FULL_SCALES, NO_TTA),
    'full        (scales x transTTA)  ': (FULL_SCALES, FULL_TTA),
}

@torch.no_grad()
def eval_tta(model, idx, scales, tta):
    model.eval(); correct = 0
    for s0 in range(0, len(idx), 64):     # small batch: 512x220 w1.3 thrashes MPS at large batch
        bi = idx[s0:s0 + 64]
        xr = XTR[bi].to(device).float() / 255.
        p = torch.zeros(len(bi), 5, device=device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            for sH, sW in scales:
                xs = xr if (sH, sW) == (IMG_H, IMG_W) else F.interpolate(xr, size=(sH, sW), mode='bilinear', align_corners=False)
                xs = (xs - MEAN_d) / STD_d
                for fs, ts in tta:
                    x = torch.roll(xs, shifts=(fs, ts), dims=(2, 3))
                    p += torch.softmax(model(x).float(), 1)
        correct += (p.argmax(1).cpu() == YTR[bi]).sum().item()
    return 100 * correct / len(idx)

# ----------------------------- run -----------------------------
results = {name: [] for name in CONFIGS}
for f, (_, vai) in enumerate(splits):
    vai = torch.as_tensor(vai)
    st = torch.load(OUT_DIR / f'best_{ROUND}f{f}.pt', map_location=device)
    m = EffNet(5, DROPOUT, DROP_PATH, width=WIDTH, depth_mult=DEPTH).to(device)
    m.load_state_dict(st)
    line = f'fold{f} (val {len(vai)}): '
    for name, (scales, tta) in CONFIGS.items():
        acc = eval_tta(m, vai, scales, tta); results[name].append(acc)
        line += f'{name.split("(")[0].strip()}={acc:.2f}  '
    print(line, flush=True)

print('\n===== TTA PROBE (val-fold mean, leakage-free) =====', flush=True)
base = float(np.mean(results['base        (448, no TTA)        ']))
for name, accs in results.items():
    mean = float(np.mean(accs)); d = mean - base
    tag = '  (anchor ~= printed CV)' if name.startswith('base') else f'  Δ {d:+.2f} vs base'
    print(f'  {name} {mean:6.2f}%{tag}', flush=True)
print(f'\nINTERPRET: "full Δ" is the real submission-time TTA gain on val. Add it to FINAL CV, then +4 native', flush=True)
print('offset -> Kaggle estimate. If +multiscale < +trans, the 512 upscale is hurting -> drop it & re-probe', flush=True)
print('with SCALES={384,448}. A negative full Δ means submit single-scale 448 (no multi-scale).', flush=True)

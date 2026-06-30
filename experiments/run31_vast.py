"""
Run 31 — MAXIMIZE ENSEMBLE MEMBERS (single CUDA GPU). 4-fold at the confirmed recipe. Real shot at 0.80.

Context: Run 30 (2-of-5-fold, 448, w1.3) = Kaggle 0.78763 — DEAD FLAT vs Run 28 (3-fold, 384, w1.1) 0.78836 —
despite CV 77.00 vs 74.77. The CV LIED: it is NOT comparable across fold schemes (Run 30's 2-of-5 trains on
80% data -> inflates val, but ensembles only 2 models -> less test smoothing; offset collapsed +4.07 -> +1.76).
README Exp 43 + memory cv-not-comparable-across-folds. KEY TAKEAWAY: the one lever that demonstrably TRANSFERS
to Kaggle is the NUMBER OF ENSEMBLE MEMBERS (it is what the val->Kaggle offset partly captures). Run 30
sabotaged its own Kaggle by ensembling only 2 models.

Run 31 leans into that: same confirmed recipe, MORE members.
  1. 4-FOLD (n_splits=4, train all 4) -> 4 ensemble members (vs Run 28's 3, Run 30's 2). Each model trains on
     75% data. More members = better test generalization = the lever that actually moves Kaggle.
  2. resolution 448x192, width 1.3 (kept — neutral-or-better on test; trained fine; no reason to drop).
  3. translation TTA ONLY at inference — multi-scale {384,448,512} FALSIFIED (Exp 43: -1.88pp val; the
     global-pool head accepts any size but features don't survive the rescale). NO multi-scale here.
  4. pseudo R1 top-500/class floor 0.50 (kept; Run 28 R1 +0.44).
NOTE: 4-fold CV is its OWN scheme — NOT comparable to Run 28's 3-fold 74.77 and the +4 offset does NOT apply.
Judge Run 31 by the KAGGLE SUBMISSION, not CV (lesson from Run 30). Expectation: 4 members should beat 3 ->
push past 0.788 toward 0.80.

⚠️ TIME BUDGET (10h cap). 4-fold + pseudo R1 = 8 fold-trainings. Run 30 was 4 (2-fold x 2 rounds), so this
is 2x. GUARDED EARLY STOP is ON (ES_PATIENCE=6, ES_GUARD=0.65): each fold stops once val plateaus in the
anneal tail (Run 30 peaked ~ep38/58, so this typically saves ~10-14 ep/fold = ~20% wall time) WITHOUT
truncating OneCycle's productive descent — best_state is always kept, so nothing found is lost. WATCH THE
FIRST FOLD'S WALL TIME (printed): if 8 folds still project > 10h, set EPOCHS=45 (recompresses the schedule —
better than early-stopping a 58-schedule) or N_ROUNDS=0 (drop pseudo, ~halves it). All knobs are env vars:
TRAIN_FOLDS, N_SPLITS, EPOCHS, N_ROUNDS, ES_PATIENCE (0=off), ES_GUARD. Safe fallback: TRAIN_FOLDS=3.

Setup (on the box):
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121   # if torch missing
    pip install -q kagglehub pandas scikit-learn pillow numpy
    mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    #   (or: export KAGGLE_USERNAME=... ; export KAGGLE_KEY=...)
Run (survives disconnect):
    tmux new -s run31
    DATA_PATH=/root/comp/data OUT_DIR=./run31_out N_ROUNDS=1 python run31_vast.py 2>&1 | tee run31.log
    # detach: Ctrl-b then d ;  reattach: tmux attach -t run31
    # if first fold is slow: add EPOCHS=45  or  N_ROUNDS=0  (or TRAIN_FOLDS=3) to fit 10h.
Result: $OUT_DIR/submission.csv  (scp it back) — already 448 + translation-TTA-only, upload directly.
"""
import os, glob, copy, math, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold

# ----------------------------- env / device -----------------------------
OUT_DIR = Path(os.environ.get('OUT_DIR', './run31_out')); OUT_DIR.mkdir(parents=True, exist_ok=True)
N_ROUNDS = int(os.environ.get('N_ROUNDS', '1'))   # 0 = folds+TTA only. 1 = +pseudo R1 (Run 28 R1 +0.44).
# GUARDED early stop (saves the dead tail without truncating OneCycle's anneal). Stop a fold only if val hasn't
# improved for ES_PATIENCE epochs AND we're already in the anneal tail (ep >= ES_GUARD*epochs). best_state is
# always the best checkpoint, so nothing found is ever lost. ES_PATIENCE=0 disables. Run 30 peaked ~ep38/58.
ES_PATIENCE = int(os.environ.get('ES_PATIENCE', '6'))    # epochs of no val gain before stopping
ES_GUARD = float(os.environ.get('ES_GUARD', '0.65'))     # earliest stop point as a fraction of total epochs
# Run 31: native-aspect 448x192 (H/W=2.33 kept). Inference is SINGLE-scale 448 (multi-scale falsified, Exp 43).
# Cheaper alt if the 10h budget is tight: IMG_H,IMG_W = 416,179 (or use EPOCHS=45 / N_ROUNDS=0).
IMG_H, IMG_W = 448, 192
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
use_amp = (device.type == 'cuda')
NGPU = torch.cuda.device_count()
torch.backends.cudnn.benchmark = True             # fixed input size -> Ampere autotune

def make_scaler():                                # version-safe AMP scaler (new torch.amp vs old torch.cuda.amp)
    try: return torch.amp.GradScaler('cuda', enabled=use_amp)
    except (AttributeError, TypeError): return torch.cuda.amp.GradScaler(enabled=use_amp)
print('torch', torch.__version__, '| device:', device, '| GPUs:', NGPU,
      '|', [torch.cuda.get_device_name(i) for i in range(NGPU)] if use_amp else 'CPU',
      '| OUT_DIR:', OUT_DIR, '| N_ROUNDS:', N_ROUNDS, flush=True)

# ----------------------------- data: local DATA_PATH (preferred), else kagglehub -----------------------------
# DATA_PATH = a folder that contains train.csv/test.csv + train/ test/ (e.g. unzipped `kaggle competitions
# download`). Use this to bypass kagglehub entirely when the image ships a broken kagglehub/kagglesdk.
_dp = os.environ.get('DATA_PATH')
if _dp:
    DATA_DIR = Path(_dp); print('using DATA_PATH =', DATA_DIR, flush=True)
else:
    if not os.environ.get('KAGGLE_KEY'):
        kj = Path.home() / '.kaggle' / 'kaggle.json'
        if not kj.exists():
            raise SystemExit('No Kaggle credentials. Put kaggle.json in ~/.kaggle/ (chmod 600) '
                             'or set KAGGLE_USERNAME / KAGGLE_KEY env vars, or set DATA_PATH to a local data folder.')
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
print('DATA_DIR =', DATA_DIR, '| TRAIN', len(glob.glob(str(TRAIN_DIR) + '/*')),
      '| TEST', len(glob.glob(str(TEST_DIR) + '/*')), flush=True)
assert TRAIN_DIR.is_dir() and TEST_DIR.is_dir()

train_df = pd.read_csv(DATA_DIR / 'train.csv')
test_df  = pd.read_csv(DATA_DIR / 'test.csv')

# per-channel mean/std from a sample (RGB spectrograms != ImageNet stats)
_smp = train_df['id'].sample(800, random_state=0); _acc = np.zeros(3); _acc2 = np.zeros(3); _n = 0
for fn in _smp:
    a = np.asarray(Image.open(TRAIN_DIR / fn).convert('RGB').resize((IMG_W, IMG_H)), np.float32) / 255.  # PIL=(W,H)
    _acc += a.reshape(-1, 3).sum(0); _acc2 += (a.reshape(-1, 3) ** 2).sum(0); _n += IMG_H * IMG_W
MEAN = torch.tensor(_acc / _n, dtype=torch.float32).view(3, 1, 1)
STD  = torch.tensor(np.sqrt(_acc2 / _n - (_acc / _n) ** 2), dtype=torch.float32).view(3, 1, 1)
print('RGB mean', MEAN.flatten().tolist(), 'std', STD.flatten().tolist(), flush=True)

def cache_u8(df, img_dir):
    X = torch.empty(len(df), 3, IMG_H, IMG_W, dtype=torch.uint8)
    for i, row in enumerate(df.itertuples(index=False)):
        a = np.asarray(Image.open(img_dir / row.id).convert('RGB').resize((IMG_W, IMG_H)))  # PIL=(W,H)
        X[i] = torch.from_numpy(a.copy()).permute(2, 0, 1)
        if (i + 1) % 3000 == 0: print(f'  {i+1}/{len(df)}', flush=True)
    return X
print('caching train...', flush=True); XTR = cache_u8(train_df, TRAIN_DIR)
print('caching test...',  flush=True); XTE = cache_u8(test_df,  TEST_DIR)
YTR = torch.tensor(train_df['label'].values - 1)
print('train', tuple(XTR.shape), f'{XTR.nelement()/1e9:.1f}GB uint8 | test', tuple(XTE.shape), flush=True)

# ----------------------------- model (from-scratch EfficientNet/MBConv, primitives only) -----------------------------
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
        for m in s.modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1: nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear) and m.bias is not None: nn.init.zeros_(m.bias)
    def forward(s, x): return s.head(s.blocks(s.stem(x)))

_m = EffNet(width=1.3, depth_mult=1.2); _n = sum(p.numel() for p in _m.parameters())
print(f'EffNet B2-scale params: {_n:,} ({_n/1e6:.2f}M)', flush=True); del _m

class EMA:
    def __init__(s, m, d=0.999): s.d = d; s.sh = copy.deepcopy(m).eval(); [p.requires_grad_(False) for p in s.sh.parameters()]
    @torch.no_grad()
    def update(s, m):
        for a, b in zip(s.sh.state_dict().values(), m.state_dict().values()):
            if a.dtype.is_floating_point: a.mul_(s.d).add_(b, alpha=1 - s.d)
            else: a.copy_(b)

# ----------------------------- train one fold (count-preserving aug + optional pseudo) -----------------------------
CONFIG = dict(epochs=int(os.environ.get('EPOCHS', '58')), batch_size=48, max_lr=1.0e-3, weight_decay=1e-3,
              label_smoothing=0.1, dropout=0.4, drop_path=0.2, ema_decay=0.999,
              width=1.3, depth_mult=1.2, ema_reset_epoch=3, pct_start=0.2)  # Run 30: KEEP w1.3 — at 384 w1.3==w1.1 (tie), but 448 raises resolution so COMPOUND SCALING says capacity should now pay (untested interaction); w1.3 at worst ties
MEAN_d = MEAN.to(device); STD_d = STD.to(device)

def gpu_aug(x):     # COUNT-PRESERVING only: translation + contrast + SNR-varying noise. NO masking/cutout (Exp 35).
    fs = int(torch.randint(-18, 19, (1,)).item()); ts = int(torch.randint(-28, 29, (1,)).item())
    x = torch.roll(x, shifts=(fs, ts), dims=(2, 3))
    if fs > 0: x[:, :, :fs, :] = 0
    elif fs < 0: x[:, :, fs:, :] = 0
    if ts > 0: x[:, :, :, :ts] = 0
    elif ts < 0: x[:, :, :, ts:] = 0
    B = x.size(0)
    c = 0.8 + 0.4 * torch.rand(B, 1, 1, 1, device=x.device); b = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.1
    sig = 0.01 + 0.05 * torch.rand(B, 1, 1, 1, device=x.device)
    x = (x - 0.5) * c + 0.5 + b + torch.randn_like(x) * sig
    return x.clamp_(0, 1)

@torch.no_grad()
def evaluate_cache(model, idx):     # always evaluates on REAL train cache -> leakage-free CV
    model.eval(); correct = 0
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for s0 in range(0, len(idx), 256):
            bi = idx[s0:s0 + 256]
            x = (XTR[bi].to(device).float() / 255. - MEAN_d) / STD_d
            correct += (model(x).float().argmax(1).cpu() == YTR[bi]).sum().item()
    return 100 * correct / len(idx)

def train_fold(tr_idx, va_idx, cfg, tag='fold0', seed=42, xext=None, yext=None, pw=0.5):
    torch.manual_seed(seed)
    bs = cfg['batch_size'] * max(1, NGPU)
    core = EffNet(5, cfg['dropout'], cfg['drop_path'], width=cfg['width'], depth_mult=cfg['depth_mult']).to(device)
    model = nn.DataParallel(core) if NGPU > 1 else core
    ema = EMA(core, cfg['ema_decay']); crit = nn.CrossEntropyLoss(label_smoothing=cfg['label_smoothing'], reduction='none')
    opt = torch.optim.AdamW(model.parameters(), lr=cfg['max_lr'], weight_decay=cfg['weight_decay'])
    scaler = make_scaler()
    tr_idx = torch.as_tensor(tr_idx); va_idx = torch.as_tensor(va_idx)
    Xf = XTR[tr_idx]; yf = YTR[tr_idx]; wf = torch.ones(len(tr_idx)); npseudo = 0
    if xext is not None and len(xext) > 0:
        npseudo = len(xext); Xf = torch.cat([Xf, xext]); yf = torch.cat([yf, yext]); wf = torch.cat([wf, torch.full((npseudo,), pw)])
    steps = len(Xf) // bs
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg['max_lr'], total_steps=cfg['epochs'] * steps, pct_start=cfg['pct_start'])
    print(f'[{tag}] {len(tr_idx)} real + {npseudo} pseudo / {len(va_idx)} val | {steps} batches/ep | {NGPU} GPU, batch {bs}', flush=True)
    best = 0.0; best_state = None; since_best = 0
    es_guard = int(ES_GUARD * cfg['epochs'])     # only allow early-stop in the anneal tail (don't truncate OneCycle's productive descent)
    for ep in range(cfg['epochs']):
        if ep == cfg['ema_reset_epoch']: ema = EMA(core, cfg['ema_decay'])
        model.train(); tc = tn = 0
        perm = torch.randperm(len(Xf))
        for s0 in range(0, steps * bs, bs):
            bi = perm[s0:s0 + bs]
            x = (gpu_aug(Xf[bi].to(device, non_blocking=True).float() / 255.) - MEAN_d) / STD_d
            yb = yf[bi].to(device, non_blocking=True); wb = wf[bi].to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(x); loss = (crit(out, yb) * wb).sum() / wb.sum()
            scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step(); ema.update(core)
            tc += (out.float().argmax(1) == yb).sum().item(); tn += yb.size(0)
        ta = 100 * tc / tn; rv = evaluate_cache(core, va_idx); ev = evaluate_cache(ema.sh, va_idx)
        acc = max(rv, ev); s = 'EMA' if ev >= rv else 'raw'
        print(f'[{tag}] ep{ep+1}/{cfg["epochs"]} train {ta:.2f} raw {rv:.2f} ema {ev:.2f} -> val {acc:.2f} ({s}) gap {ta-acc:+.1f} lr {opt.param_groups[0]["lr"]:.2e}', flush=True)
        if acc > best:
            best = acc; since_best = 0; chosen = ema.sh if ev >= rv else core
            best_state = {k: v.detach().cpu().clone() for k, v in chosen.state_dict().items()}
            torch.save(best_state, OUT_DIR / f'best_{tag}.pt')
        else:
            since_best += 1
            # guarded early stop: only in the anneal tail (ep>=ES_GUARD*epochs) AND no gain for ES_PATIENCE epochs.
            if ES_PATIENCE and (ep + 1) >= es_guard and since_best >= ES_PATIENCE:
                print(f'[{tag}] early stop @ ep{ep+1}/{cfg["epochs"]} (no gain {since_best} ep, best {best:.2f} @ ep{ep+1-since_best})', flush=True)
                break
    print(f'[{tag}] BEST {best:.2f}%', flush=True); return best, best_state

# ----------------------------- 2-of-5-fold ensemble + aggressive top-K pseudo-labeling (leakage-free CV gate) -----------------------------
# Run 31: 4-WAY split, train ALL 4 -> 4 ensemble members (the lever that transfers to Kaggle). Each model
# sees 75% data. Pseudo: top-500/class, floor 0.50. All knobs env-configurable for the 10h time budget.
N_SPLITS = int(os.environ.get('N_SPLITS', '4')); TRAIN_FOLDS = int(os.environ.get('TRAIN_FOLDS', '4'))
TOPK = 500; CONF_FLOOR = 0.50; PW = 0.5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
splits = list(skf.split(np.arange(len(train_df)), train_df['label']))[:TRAIN_FOLDS]

def predict_test_probs(states):
    probs = torch.zeros(len(test_df), 5)
    for st in states:
        m = EffNet(5, CONFIG['dropout'], CONFIG['drop_path'], width=CONFIG['width'], depth_mult=CONFIG['depth_mult']).to(device)
        m.load_state_dict(st); m.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            for s0 in range(0, len(test_df), 256):
                x = (XTE[s0:s0 + 256].to(device).float() / 255. - MEAN_d) / STD_d
                probs[s0:s0 + 256] += torch.softmax(m(x).float(), 1).cpu()
    return probs / len(states)

def build_pseudo(probs):     # top-K most-confident per predicted class (NOT an absolute threshold — Exp 37)
    conf, lab = probs.max(1); sel = []
    for c in range(5):
        ci = torch.where(lab == c)[0]
        ci = ci[conf[ci].argsort(descending=True)]
        ci = ci[conf[ci] >= CONF_FLOOR][:TOPK]
        sel.append(ci)
    per = [len(s) for s in sel]
    sel = torch.cat(sel) if sel else torch.tensor([], dtype=torch.long)
    rng = f'[{conf[sel].min():.3f},{conf[sel].max():.3f}]' if len(sel) else '[n/a]'
    print(f'pseudo: top-{TOPK}/class floor{CONF_FLOOR} -> per-class {per}, {len(sel)} used; conf range {rng}', flush=True)
    return sel, lab[sel]

def run_round(xext, yext, tagp):
    states, accs = [], []
    for f, (tri, vai) in enumerate(splits):
        t0 = time.time()
        b, st = train_fold(tri, vai, CONFIG, tag=f'{tagp}f{f}', seed=42 + f, xext=xext, yext=yext, pw=PW)
        dt = (time.time() - t0) / 60
        accs.append(b); states.append(st)
        n_total = TRAIN_FOLDS * (N_ROUNDS + 1)   # 10h budget watch: extrapolate from this fold's wall time
        print(f'[{tagp}f{f}] fold time {dt:.1f} min | est total {n_total} folds = {dt*n_total/60:.1f} h '
              f'({"OK <10h" if dt*n_total/60 <= 10 else "OVER 10h -> set EPOCHS=45 or N_ROUNDS=0"})', flush=True)
    cv = float(np.mean(accs)); print(f'[{tagp}] CV {cv:.2f}%  folds {[round(a,1) for a in accs]}', flush=True)
    return cv, states

cv0, fold_states = run_round(None, None, 'R0')
best_cv = cv0
for r in range(1, N_ROUNDS + 1):
    sel, plab = build_pseudo(predict_test_probs(fold_states))
    if len(sel) == 0:
        print('no pseudo selected -> stop'); break
    cv, states = run_round(XTE[sel], plab, f'R{r}')
    if cv > best_cv:                                  # adopt on ANY improvement (gate lowered from +0.5 per request):
        print(f'R{r} PROMOTED ({cv:.2f} > {best_cv:.2f}) -> new best, applies to submission', flush=True)
        best_cv = cv; fold_states = states            # fold_states now always holds the best-CV round
    else:                                             # no gain: keep best-so-far (already in fold_states) & stop
        print(f'R{r} no gain ({cv:.2f} <= {best_cv:.2f}) -> keep best round R{r-1}', flush=True); break
print(f'\nFINAL CV {best_cv:.2f}%  ({TRAIN_FOLDS}-fold scheme) -> best round applied', flush=True)
print('NOTE: the +4 native offset was calibrated on 3-FOLD and does NOT apply to other schemes (Run 30 lesson:', flush=True)
print(f'      CV is not comparable across fold schemes). {TRAIN_FOLDS}-fold has more members -> JUDGE BY THE KAGGLE', flush=True)
print('      SUBMISSION, not this CV. Goal: 5 members beat 3 -> push past 0.788.', flush=True)

# ----------------------------- inference: ensemble + translation TTA ONLY -> submission.csv -----------------------------
# Run 31: single-scale 448 + translation TTA. Multi-scale {384,448,512} was FALSIFIED on val (Exp 43,
# tta_probe.py: -1.88pp — the global-pool head accepts any size but features don't survive the rescale).
# Translation TTA verified +0.44pp. NO multi-scale, NO flips/rotation (flips harmful, Exp 6).
TTA = [(0, 0), (0, 12), (0, -12), (6, 0), (-6, 0)]   # translation TTA only (valid; verified +0.44)
probs = torch.zeros(len(test_df), 5)
for st in fold_states:
    m = EffNet(5, CONFIG['dropout'], CONFIG['drop_path'], width=CONFIG['width'], depth_mult=CONFIG['depth_mult']).to(device)
    m.load_state_dict(st); m.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        for s0 in range(0, len(test_df), 256):
            xs = (XTE[s0:s0 + 256].to(device).float() / 255. - MEAN_d) / STD_d   # native 448x192, no rescale
            for fs, ts in TTA:
                x = torch.roll(xs, shifts=(fs, ts), dims=(2, 3))
                probs[s0:s0 + 256] += torch.softmax(m(x).float(), 1).cpu()
print(f'inference: {len(fold_states)} folds x 1 scale (448) x {len(TTA)} TTA = {len(fold_states)*len(TTA)} passes/image', flush=True)
pred = probs.argmax(1).numpy() + 1
sub = pd.DataFrame({'id': test_df['id'].tolist(), 'label': pred})
sub_path = OUT_DIR / 'submission.csv'; sub.to_csv(sub_path, index=False)
print(sub['label'].value_counts().sort_index(), flush=True)
print('-> wrote', sub_path, flush=True)

"""
Run 27 — count-aware EffNet-B2 + top-K pseudo-labeling, adapted for a SINGLE CUDA GPU (vast.ai / RTX A5000).

Standalone port of notebooks/07_colab_effnet_b2.ipynb. Identical model / aug / training / pseudo-label
logic; only the Colab-specific bits changed: Kaggle creds via env or ~/.kaggle/kaggle.json (no Colab
secrets), outputs to OUT_DIR, single-GPU (no DataParallel), cudnn.benchmark on for Ampere.

Setup (on the box):
    pip install -q torch --index-url https://download.pytorch.org/whl/cu121   # if torch missing
    pip install -q kagglehub pandas scikit-learn pillow numpy
    mkdir -p ~/.kaggle && cp kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
    #   (or: export KAGGLE_USERNAME=... ; export KAGGLE_KEY=...)
Run (survives disconnect):
    tmux new -s run27
    DATA_PATH=/root/comp/data OUT_DIR=./run27_out N_ROUNDS=2 python run27_vast.py 2>&1 | tee run27.log
    # detach: Ctrl-b then d ;  reattach: tmux attach -t run27
Result: $OUT_DIR/submission.csv  (scp it back).
"""
import os, glob, copy, math
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.model_selection import StratifiedKFold

# ----------------------------- env / device -----------------------------
OUT_DIR = Path(os.environ.get('OUT_DIR', './run27_out')); OUT_DIR.mkdir(parents=True, exist_ok=True)
N_ROUNDS = int(os.environ.get('N_ROUNDS', '2'))   # 0 = 3-fold+TTA only (~77). 1 = pseudo push (validated +1.5pp).
# NATIVE ASPECT (README Exp 41): native spectrogram is 128(freq)x55(time), H/W=2.327. Square-stretching
# to 256x256 smeared adjacent objects along time and cost +1.77pp (confirmed at ~50k px). Use native
# aspect at a matched ~63k-px budget so this is a clean apples-to-apples upgrade vs the 256^2 CV 71.34.
# Cheaper alt (less compute, similar gain expected): IMG_H, IMG_W = 256, 112.
IMG_H, IMG_W = 384, 165
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

_m = EffNet(width=1.1, depth_mult=1.2); _n = sum(p.numel() for p in _m.parameters())
print(f'EffNet B2-scale params: {_n:,} ({_n/1e6:.2f}M)', flush=True); del _m

class EMA:
    def __init__(s, m, d=0.999): s.d = d; s.sh = copy.deepcopy(m).eval(); [p.requires_grad_(False) for p in s.sh.parameters()]
    @torch.no_grad()
    def update(s, m):
        for a, b in zip(s.sh.state_dict().values(), m.state_dict().values()):
            if a.dtype.is_floating_point: a.mul_(s.d).add_(b, alpha=1 - s.d)
            else: a.copy_(b)

# ----------------------------- train one fold (count-preserving aug + optional pseudo) -----------------------------
CONFIG = dict(epochs=58, batch_size=48, max_lr=1.0e-3, weight_decay=1e-3,
              label_smoothing=0.1, dropout=0.4, drop_path=0.2, ema_decay=0.999,
              width=1.1, depth_mult=1.2, ema_reset_epoch=3, pct_start=0.2)  # Run 28 config (native 384x165, w1.1) -> Kaggle 0.78836. Run 29 (w1.3) is in run29_vast.py.
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
    best = 0.0; best_state = None
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
            best = acc; chosen = ema.sh if ev >= rv else core
            best_state = {k: v.detach().cpu().clone() for k, v in chosen.state_dict().items()}
            torch.save(best_state, OUT_DIR / f'best_{tag}.pt')
    print(f'[{tag}] BEST {best:.2f}%', flush=True); return best, best_state

# ----------------------------- 3-fold ensemble + top-K pseudo-labeling (leakage-free CV gate) -----------------------------
N_FOLDS = 3; TOPK = 350; CONF_FLOOR = 0.55; PW = 0.5
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
splits = list(skf.split(np.arange(len(train_df)), train_df['label']))

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
        b, st = train_fold(tri, vai, CONFIG, tag=f'{tagp}f{f}', seed=42 + f, xext=xext, yext=yext, pw=PW)
        accs.append(b); states.append(st)
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
print(f'\nFINAL CV {best_cv:.2f}%  (Kaggle est ~{best_cv+2.7:.1f}%) -> best round applied', flush=True)

# ----------------------------- inference: ensemble + translation TTA -> submission.csv -----------------------------
probs = torch.zeros(len(test_df), 5)
for st in fold_states:
    m = EffNet(5, CONFIG['dropout'], CONFIG['drop_path'], width=CONFIG['width'], depth_mult=CONFIG['depth_mult']).to(device)
    m.load_state_dict(st); m.eval()
    with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        TTA = [(0, 0), (0, 12), (0, -12), (6, 0), (-6, 0)]   # translation TTA only (valid; no flips/rot)
        for s0 in range(0, len(test_df), 256):
            x0 = (XTE[s0:s0 + 256].to(device).float() / 255. - MEAN_d) / STD_d
            for fs, ts in TTA:
                x = torch.roll(x0, shifts=(fs, ts), dims=(2, 3))
                probs[s0:s0 + 256] += torch.softmax(m(x).float(), 1).cpu()
pred = probs.argmax(1).numpy() + 1
sub = pd.DataFrame({'id': test_df['id'].tolist(), 'label': pred})
sub_path = OUT_DIR / 'submission.csv'; sub.to_csv(sub_path, index=False)
print(sub['label'].value_counts().sort_index(), flush=True)
print('-> wrote', sub_path, flush=True)

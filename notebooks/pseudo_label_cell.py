# =============================================================================
# Phase 4 — Semi-supervised pseudo-labeling (path-to-80 campaign, Exp 35).
# Reference cell to paste into 07_kaggle_effnet_b2.ipynb AFTER the recipe is locked
# (winning stem/aug from Phase 2+3). The test set may be used (user-confirmed).
#
# Idea: a 3-fold ensemble at ~75% can label the 5,000 test images; the high-confidence
# ones become extra training data that directly targets the test distribution. Classic
# +2-4pp lever. Validated on LEAKAGE-FREE clean folds (pseudo-labels are test-only,
# disjoint from every val fold) BEFORE spending a submission.
#
# Assumes these already exist from the notebook:
#   XTR [Ntr,3,IMG,IMG] uint8, YTR [Ntr] (0-idx), XTE [Nte,3,IMG,IMG] uint8
#   MEAN_d, STD_d, device, use_amp, EffNet, CONFIG, gpu_aug, evaluate_cache
#   train_fold(tr_idx, va_idx, cfg, tag, seed) -> (best_acc, best_state)
# train_fold must be extended to accept optional extra data (see PATCH below).
# =============================================================================
import numpy as np, torch
from sklearn.model_selection import StratifiedKFold

TAU = 0.95          # confidence threshold (start HIGH; never lower across rounds)
PSEUDO_W = 0.5      # loss weight for pseudo samples vs real (1.0)
PSEUDO_SMOOTH = 0.2 # extra label smoothing on pseudo labels (hedge against wrong labels)
N_ROUNDS = 2        # hard stop


# ---- PATCH train_fold to mix in extra (pseudo) samples ----------------------
# In train_fold, accept (Xext, yext, w_ext) and, each batch, draw a fraction of pseudo
# samples; apply CrossEntropy with per-sample weights and extra smoothing on the pseudo part.
# Minimal change: build a combined index space with a boolean is_pseudo mask, and in the
# loss use F.cross_entropy(out, yb, reduction='none') * sample_w, where sample_w = 1.0 for
# real and PSEUDO_W for pseudo, and label_smoothing handled by mixing two CE terms.
# (Keep EMA/OneCycle/eval identical; eval stays on REAL val folds only.)


def predict_test_probs(fold_states, cfg):
    """Fold-averaged softmax over the test cache (no TTA, for honest confidence)."""
    probs = torch.zeros(len(XTE), 5)
    for st in fold_states:
        m = EffNet(5, cfg['dropout'], cfg['drop_path'], width=cfg['width'],
                   depth_mult=cfg['depth_mult']).to(device)
        m.load_state_dict(st); m.eval()
        with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            for s0 in range(0, len(XTE), 256):
                x = (XTE[s0:s0+256].to(device).float()/255. - MEAN_d)/STD_d
                probs[s0:s0+256] += torch.softmax(m(x).float(), 1).cpu()
    return probs / len(fold_states)


def build_pseudo(probs):
    """High-confidence, class-capped pseudo-labels to prevent count drift."""
    conf, lab = probs.max(1)
    keep = conf > TAU
    idx = torch.where(keep)[0]
    lab_k, conf_k = lab[idx], conf[idx]
    # class cap at the median per-class count (drop lowest-confidence excess)
    per = [ (lab_k == c).sum().item() for c in range(5) ]
    cap = int(np.median([p for p in per if p > 0]) or 0)
    sel = []
    for c in range(5):
        ci = idx[lab_k == c]
        cc = conf_k[lab_k == c]
        order = ci[cc.argsort(descending=True)][:cap]
        sel.append(order)
    sel = torch.cat(sel) if sel else torch.tensor([], dtype=torch.long)
    print(f'  pseudo: kept {len(idx)}/{len(XTE)} > {TAU}; class-capped to {cap}/class -> {len(sel)} used')
    print(f'  pseudo class dist: {[int((lab[sel]==c).sum()) for c in range(5)]}')
    return sel, lab[sel]


# ---- main loop --------------------------------------------------------------
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
splits = list(skf.split(np.arange(len(YTR)), YTR.numpy()))

def run_round(extra=None, tag=''):
    states, accs = [], []
    for f, (tri, vai) in enumerate(splits):
        # extra=(Xext,yext) pseudo data is appended to THIS fold's training set only;
        # val fold (vai) stays pure real labels -> leakage-free CV.
        b, st = train_fold(tri, vai, CONFIG, tag=f'{tag}f{f}', seed=42+f)  # + extra args after PATCH
        accs.append(b); states.append(st)
    cv = float(np.mean(accs)); print(f'[{tag}] CV mean {cv:.2f}%  folds {["%.1f"%a for a in accs]}')
    return cv, states

cv0, states = run_round(tag='R0')                       # real labels only
best_cv, best_states = cv0, states
for r in range(1, N_ROUNDS+1):
    probs = predict_test_probs(states, CONFIG)
    sel, plab = build_pseudo(probs)
    Xext, yext = XTE[sel], plab                          # pseudo training data (test-only)
    cv, states = run_round(extra=(Xext, yext), tag=f'R{r}')
    if cv >= best_cv + 0.5:                              # GATE: clean-fold CV must improve
        print(f'  round {r} PROMOTED ({cv:.2f} >= {best_cv:.2f}+0.5)'); best_cv, best_states = cv, states
    else:
        print(f'  round {r} rejected ({cv:.2f} < {best_cv:.2f}+0.5) -> stop, revert'); break

print(f'\nFINAL pseudo-label CV {best_cv:.2f}% (R0 was {cv0:.2f}%). Use best_states for submission.')

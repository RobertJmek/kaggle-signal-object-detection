"""
Phase 4 OFFLINE VALIDATION — does pseudo-labeling actually lift accuracy? (path-to-82)

All structural levers failed their gates (ordinal head Exp 33, masking Exp 35, Gabor stem Exp 35).
Pseudo-labeling is the single biggest UNVALIDATED lever left. Validate it offline, decisively, before
spending it on a Colab run: carve a stratified HOLD slice out of train as a proxy "test" whose true
labels are KNOWN but HIDDEN during training (revealed only to score). Then run the real self-training
loop and read the gain on BOTH a clean real-val fold (the leakage-free CV gate the notebook uses) and
the proxy test's true accuracy (the number we actually care about).

  R0: train on TR (real labels) -> measure va_acc (clean) + test_acc (proxy true) + test softmax probs
  R1: pseudo-label the proxy test (conf>TAU, class-capped), retrain on TR + pseudo(weight PW) from
      scratch -> measure va_acc + test_acc again.
  Delta(R0->R1) on test_acc is the honest expected pseudo gain (a FLOOR: proxy test is iid from train,
  so this captures the data/self-training mechanism but not any extra Kaggle distribution-shift gain).

Identical backbone/aug/sched to the winning recipe (EffNet B0-MBConv @160, count-preserving gpu_aug,
softmax head, EMA, OneCycle). Run from src/:  python3 ab_pseudo.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ab_ordinal_test import (EffNet, EMA, S, BATCH, MAX_LR, WD, SEED, device, cache, gpu_aug)

EPOCHS = 32
HOLD = 3500          # proxy-test size (stratified), labels hidden during training
TAU = 0.85           # pseudo confidence threshold. NOTE: training uses label_smoothing=0.1, which caps
                     # realized max-softmax at ~0.92 -> TAU=0.95 selects ~0 images (the notebook's bug).
PW = 0.5             # pseudo loss weight vs real (1.0)
PSEUDO_SMOOTH = 0.2  # extra label smoothing on pseudo labels
TOPK = 250           # pseudo: top-K most-confident per predicted class (class-balanced, ~1250 total)
CONF_FLOOR = 0.55    # loose floor (p50~0.49) so we never inject near-random predictions


def train_model(Xtr, ytr, wtr, Xva, yva, mean, std, tag, smooth_extra=None, n_real=None):
    """Train EffNet softmax. wtr = per-sample loss weight. smooth_extra/n_real apply higher label
    smoothing to the pseudo tail (indices >= n_real). Returns (best_va_acc, best_ema_state)."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = EffNet(5).to(device); ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    N = len(Xtr); steps = N // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=EPOCHS * steps, pct_start=0.2)
    md = mean.to(device); sd = std.to(device)
    wtr = wtr.to(device)
    print(f'\n[{tag}] {sum(p.numel() for p in model.parameters())/1e6:.2f}M | N={N} steps={steps}', flush=True)

    @torch.no_grad()
    def evaluate(m, Xset, yset):
        m.eval(); ex = 0
        for s0 in range(0, len(Xset), 256):
            xb = (Xset[s0:s0+256].to(device).float()/255. - md)/sd
            ex += (m(xb).argmax(1).cpu() == yset[s0:s0+256]).sum().item()
        return 100*ex/len(Xset)

    best = 0.0; best_state = None; t0 = time.time()
    for ep in range(EPOCHS):
        if ep == 3: ema = EMA(model, 0.999)
        model.train()
        perm = torch.randperm(N)
        for s0 in range(0, steps*BATCH, BATCH):
            bi = perm[s0:s0+BATCH]
            xb = (gpu_aug(Xtr[bi].to(device).float()/255.) - md)/sd
            yb = ytr[bi].to(device); wb = wtr[bi]
            opt.zero_grad(set_to_none=True)
            out = model(xb)
            if smooth_extra is not None and n_real is not None:
                ls = torch.where(bi.to(device) >= n_real, torch.tensor(smooth_extra, device=device),
                                 torch.tensor(0.1, device=device))
                # per-sample CE with per-sample label smoothing via manual soft targets
                logp = F.log_softmax(out, 1)
                nll = -logp.gather(1, yb[:, None]).squeeze(1)
                smooth = -logp.mean(1)
                ce = (1 - ls) * nll + ls * smooth
            else:
                ce = F.cross_entropy(out, yb, reduction='none', label_smoothing=0.1)
            loss = (ce * wb).sum() / wb.sum()
            loss.backward(); opt.step(); sched.step(); ema.update(model)
        ema_acc = evaluate(ema.sh, Xva, yva); raw_acc = evaluate(model, Xva, yva)
        if ema_acc > best:   # select+save on EMA ONLY so saved state == tracked best (the deploy model)
            best = ema_acc; best_state = {k: v.detach().cpu().clone() for k, v in ema.sh.state_dict().items()}
        if (ep+1) % 8 == 0:
            print(f'  [{tag}] ep{ep+1:2d}/{EPOCHS} ema {ema_acc:5.2f} raw {raw_acc:5.2f} (best ema {best:5.2f})', flush=True)
    print(f'[{tag}] BEST va {best:.2f}%  [{time.time()-t0:.0f}s]', flush=True)
    return best, best_state


@torch.no_grad()
def predict_probs(state, Xset, mean, std):
    md = mean.to(device); sd = std.to(device)
    m = EffNet(5).to(device); m.load_state_dict(state); m.eval()
    probs = torch.zeros(len(Xset), 5)
    for s0 in range(0, len(Xset), 256):
        xb = (Xset[s0:s0+256].to(device).float()/255. - md)/sd
        probs[s0:s0+256] = torch.softmax(m(xb).float(), 1).cpu()
    return probs


def build_pseudo(probs, topk=TOPK, floor=CONF_FLOOR):
    """Top-K most-confident per predicted class (class-balanced), with a soft confidence floor.
    Robust to the LS-capped low-confidence regime where absolute thresholds select ~nothing."""
    conf, lab = probs.max(1)
    sel = []
    for c in range(5):
        ci = torch.where(lab == c)[0]
        cc = conf[ci]
        ci = ci[cc.argsort(descending=True)]
        ci = ci[conf[ci] >= floor][:topk]      # take most-confident-K above a loose floor
        sel.append(ci)
    per = [len(s) for s in sel]
    sel = torch.cat(sel) if sel else torch.tensor([], dtype=torch.long)
    return sel, lab[sel], per, topk


def main():
    from sklearn.model_selection import train_test_split
    print(f'device={device} S={S} EPOCHS={EPOCHS} HOLD={HOLD} TOPK={TOPK}/class floor={CONF_FLOOR} PW={PW}', flush=True)
    X, y = cache()
    idx = np.arange(len(y))
    pool_i, test_i = train_test_split(idx, test_size=HOLD, stratify=y.numpy(), random_state=SEED)
    tr_i, va_i = train_test_split(pool_i, test_size=0.15, stratify=y.numpy()[pool_i], random_state=SEED)
    Xtr, ytr = X[tr_i], y[tr_i]
    Xva, yva = X[va_i], y[va_i]
    Xte, yte = X[test_i], y[test_i]   # yte = HIDDEN ground truth, used only to score
    px = Xtr.float()/255.
    mean = px.mean(dim=(0,2,3), keepdim=True)[0]; std = px.std(dim=(0,2,3), keepdim=True)[0]; del px
    print(f'TR={len(tr_i)} VA={len(va_i)} TEST(proxy)={len(test_i)}', flush=True)

    # ---- Round 0: real labels only ----
    w0 = torch.ones(len(Xtr))
    va0, st0 = train_model(Xtr, ytr, w0, Xva, yva, mean, std, 'R0')
    probs0 = predict_probs(st0, Xte, mean, std)
    test0 = 100 * (probs0.argmax(1) == yte).float().mean().item()
    conf0 = probs0.max(1).values
    qs = torch.quantile(conf0, torch.tensor([0.5, 0.75, 0.9, 0.95, 0.99]))
    print(f'\n[R0] clean va {va0:.2f}%  |  proxy-test true acc {test0:.2f}%', flush=True)
    print(f'[R0] max-softmax conf percentiles p50/75/90/95/99 = '
          f'{qs[0]:.3f}/{qs[1]:.3f}/{qs[2]:.3f}/{qs[3]:.3f}/{qs[4]:.3f}  (LS=0.1 caps ~0.92)', flush=True)

    # ---- Build pseudo-labels: top-K most-confident per class (robust to LS-capped low confidence) ----
    sel, plab, per, _ = build_pseudo(probs0)
    # quality of the pseudo-labels we'd actually inject (we can check because yte is known):
    p_correct = 100 * (plab == yte[sel]).float().mean().item() if len(sel) else 0.0
    print(f'[pseudo] top-{TOPK}/class (floor {CONF_FLOOR}): per-class {per} -> {len(sel)} used | '
          f'pseudo-label accuracy = {p_correct:.2f}% (vs R0 test acc {test0:.1f}% — higher = conf helps)', flush=True)

    # ---- Round 1: real + pseudo (from scratch) ----
    Xp, yp = Xte[sel], plab
    Xtr1 = torch.cat([Xtr, Xp]); ytr1 = torch.cat([ytr, yp])
    w1 = torch.cat([torch.ones(len(Xtr)), torch.full((len(Xp),), PW)])
    va1, st1 = train_model(Xtr1, ytr1, w1, Xva, yva, mean, std, 'R1',
                           smooth_extra=PSEUDO_SMOOTH, n_real=len(Xtr))
    probs1 = predict_probs(st1, Xte, mean, std)
    test1 = 100 * (probs1.argmax(1) == yte).float().mean().item()

    print('\n===== PHASE 4 PSEUDO-LABEL VALIDATION =====')
    print(f'  clean val :  R0 {va0:5.2f}%  ->  R1 {va1:5.2f}%   (Δ {va1-va0:+.2f})  [the CV gate]')
    print(f'  proxy test:  R0 {test0:5.2f}%  ->  R1 {test1:5.2f}%   (Δ {test1-test0:+.2f})  [true gain]')
    print(f'  pseudo set: {len(sel)} imgs @ {p_correct:.1f}% correct')
    verdict = 'PSEUDO HELPS — promote to Colab run' if (test1 - test0) >= 0.5 else \
              'no clear gain on iid proxy (real Kaggle shift may still help; treat as inconclusive-low)'
    print(f'  VERDICT: {verdict}')


if __name__ == '__main__':
    main()

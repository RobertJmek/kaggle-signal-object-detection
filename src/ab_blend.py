"""
Phase 5 A/B (path-to-82) — does a MULTI-ARCHITECTURE blend beat the EffNet alone?

All other levers are measured; this is the last gap-closer. Train a SECOND, deliberately distinct
from-scratch net and ensemble it with the EffNet. Diversity is the point: the EffNet is depthwise-
separable MBConv + squeeze-excite; the contrast net (`ResNetCounter`) is PLAIN 3x3 convs + residual
blocks (zero-init last BN = identity start, the Exp 16 trick) — different inductive bias => decorrelated
errors => the average can beat both. Primitives-only, self-contained (competition-legal).

Same split as Exp 33/35/38 (StratifiedKFold5 first split, seed42, S=160, softmax, count-preserving aug,
EMA, 32 ep) so the EffNet solo == 67.68 reference. Reports each solo + the blend at several weights.
Gate: blend >= best solo + 0.5pp => add the 2nd arch to the Colab ensemble.
Run from src/:  python3 ab_blend.py
"""
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ab_ordinal_test import (EffNet, EMA, S, BATCH, MAX_LR, WD, SEED, device, cache, gpu_aug)

EPOCHS = 32


# ----------------------------- contrast architecture: plain-conv ResNet (primitives only) -----------------------------
class BasicBlock(nn.Module):
    def __init__(s, cin, cout, stride=1):
        super().__init__()
        s.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False); s.b1 = nn.BatchNorm2d(cout)
        s.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False);      s.b2 = nn.BatchNorm2d(cout)
        s.act = nn.SiLU(True)
        s.short = (nn.Sequential() if (stride == 1 and cin == cout)
                   else nn.Sequential(nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout)))
        nn.init.zeros_(s.b2.weight)            # identity-start residual (Exp 16 fix)
    def forward(s, x):
        out = s.act(s.b1(s.c1(x))); out = s.b2(s.c2(out)); return s.act(out + s.short(x))


class ResNetCounter(nn.Module):
    def __init__(s, n_out=5, width=48, dropout=0.3):
        super().__init__()
        s.stem = nn.Sequential(nn.Conv2d(3, width, 3, 2, 1, bias=False), nn.BatchNorm2d(width), nn.SiLU(True))
        cfg = [(width, 2, 1), (width * 2, 2, 2), (width * 4, 2, 2), (width * 8, 2, 2)]  # (channels, nblocks, stride)
        blocks = []; cin = width
        for co, nb, st in cfg:
            for j in range(nb):
                blocks.append(BasicBlock(cin, co, st if j == 0 else 1)); cin = co
        s.blocks = nn.Sequential(*blocks)
        s.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(dropout), nn.Linear(cin, n_out))
        for m in s.modules():
            if isinstance(m, nn.Conv2d) and m.groups == 1:
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        for m in s.modules():                  # re-zero the residual BNs clobbered by the kaiming loop
            if isinstance(m, BasicBlock): nn.init.zeros_(m.b2.weight)
    def forward(s, x): return s.head(s.blocks(s.stem(x)))


def train_net(make_model, tag, X, y, tr, va, mean, std):
    """Train one model; return (best_val_acc, val_softmax_probs at best EMA state)."""
    torch.manual_seed(SEED); np.random.seed(SEED)
    model = make_model().to(device); ema = EMA(model, 0.999)
    opt = torch.optim.AdamW(model.parameters(), lr=MAX_LR, weight_decay=WD)
    tr_t = torch.as_tensor(tr); steps = len(tr_t) // BATCH
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=MAX_LR, total_steps=EPOCHS * steps, pct_start=0.2)
    md = mean.to(device); sd = std.to(device)
    print(f'\n[{tag}] {sum(p.numel() for p in model.parameters())/1e6:.2f}M | {steps} batches/ep', flush=True)

    @torch.no_grad()
    def probs_of(m):
        m.eval(); out = torch.zeros(len(va), 5)
        for s0 in range(0, len(va), 256):
            bi = va[s0:s0+256]
            xb = (X[bi].to(device).float()/255. - md)/sd
            out[s0:s0+256] = torch.softmax(m(xb).float(), 1).cpu()
        return out

    best = 0.0; best_probs = None; t0 = time.time()
    for ep in range(EPOCHS):
        if ep == 3: ema = EMA(model, 0.999)
        model.train()
        perm = tr_t[torch.randperm(len(tr_t))]
        for s0 in range(0, steps*BATCH, BATCH):
            bi = perm[s0:s0+BATCH]
            xb = (gpu_aug(X[bi].to(device).float()/255.) - md)/sd
            yb = y[bi].to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(xb), yb, label_smoothing=0.1)
            loss.backward(); opt.step(); sched.step(); ema.update(model)
        pe = probs_of(ema.sh); pr = probs_of(model)
        ae = 100*(pe.argmax(1) == y[va]).float().mean().item()
        ar = 100*(pr.argmax(1) == y[va]).float().mean().item()
        if ae >= ar and ae > best: best = ae; best_probs = pe
        elif ar > ae and ar > best: best = ar; best_probs = pr
        if (ep+1) % 8 == 0:
            print(f'  [{tag}] ep{ep+1:2d}/{EPOCHS} ema {ae:5.2f} raw {ar:5.2f} (best {best:5.2f})', flush=True)
    print(f'[{tag}] BEST solo {best:.2f}%  [{time.time()-t0:.0f}s]', flush=True)
    return best, best_probs


def main():
    from sklearn.model_selection import StratifiedKFold
    print(f'device={device} S={S} epochs={EPOCHS} (Phase 5 multi-arch blend)', flush=True)
    X, y = cache()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    tr, va = next(iter(skf.split(np.arange(len(y)), y.numpy())))
    va = torch.as_tensor(va)
    px = X[torch.as_tensor(tr)].float()/255.
    mean = px.mean(dim=(0,2,3), keepdim=True)[0]; std = px.std(dim=(0,2,3), keepdim=True)[0]; del px

    a_eff, p_eff = train_net(lambda: EffNet(5),         'EFFNET', X, y, tr, va, mean, std)
    a_res, p_res = train_net(lambda: ResNetCounter(5),  'RESNET', X, y, tr, va, mean, std)

    yv = y[va]
    print('\n===== PHASE 5 MULTI-ARCH BLEND (exact-count acc) =====')
    print(f'  EFFNET solo : {a_eff:.2f}%')
    print(f'  RESNET solo : {a_res:.2f}%')
    best_solo = max(a_eff, a_res); best_blend = 0.0; best_w = None
    for w in (0.3, 0.4, 0.5, 0.6, 0.7):                  # weight on EffNet
        blend = w*p_eff + (1-w)*p_res
        acc = 100*(blend.argmax(1) == yv).float().mean().item()
        print(f'  blend w_eff={w:.1f} : {acc:.2f}%   (Δ vs best solo {acc-best_solo:+.2f})')
        if acc > best_blend: best_blend, best_w = acc, w
    print(f'GATE (+0.5): best blend {best_blend:.2f} (w_eff={best_w}) vs best solo {best_solo:.2f} -> '
          f'{"ADD 2nd arch to ensemble" if best_blend >= best_solo + 0.5 else "no win; single arch"}')


if __name__ == '__main__':
    main()

# Signal Object Detection - Kaggle Competition

Classification of signal spectrograms into 5 categories.

## Project Structure

```
.
├── data/
│   ├── raw/          # Competition data (symlinked from ~/.cache/kagglehub)
│   └── submissions/  # Generated submission files
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_tpu_train.ipynb        # Colab GPU(T4)/TPU training (Run 20, current main pipeline)
│   ├── 03_tpu_train_large.ipynb  # Large 14.8M probe — L ≈ M (Exp 26), capacity dead; run deprioritized
│   ├── 04_tpu_train_run21_structreg.ipynb  # Run 21: structural-reg ablation (wd 3e-3 + drop_path 0.05, no MixUp)
│   ├── 05_tpu_train_hires.ipynb            # Run 23: resolution-preserving CNN (16×8, no BlurPool) — thin-line hypothesis
│   ├── 06_tpu_train_effnet224.ipynb        # Run 24: from-scratch EfficientNet/MBConv @ 224×224 RGB → 70.45% ✅
│   ├── 07_tpu_train_effnet_b2.ipynb        # Run 25: B2-scale (7.8M) @ 256×256 RGB — compound scale-up (Plan 2), Colab
│   └── 07_kaggle_effnet_b2.ipynb           # Run 25 for Kaggle (T4 ×2, DataParallel; data /kaggle/input)
├── src/
│   ├── model.py            # Custom CompactSignalCNN (PyTorch primitives only)
│   ├── train.py            # Training script (local, MPS)
│   ├── predict.py          # Inference & submission script
│   ├── ab_input_test.py        # Exp 21 A/B: G channel vs viridis-inversion
│   ├── ab_rgb_test.py          # Exp 21 A/B: G channel vs full RGB
│   ├── ab_resolution_test.py   # Exp 23 A/B: 128×64 vs 128×128 vs 224×224
│   └── ab_capacity_test.py     # Exp 25 A/B: S 1.7M vs M 7M vs L 14.8M
├── models/           # Saved model checkpoints
└── README.md
```

## Data

- **Train**: 15,000 spectrogram images in `train/` with labels 1-5
- **Test**: 5,000 spectrogram images in `test/`
- Images are grayscale spectrograms of signals

## Quick Start

### 1. Exploratory Data Analysis
```bash
cd notebooks
jupyter notebook 01_eda.ipynb
```

### 2. Train Model
```bash
cd src
python3 train.py
```

### 3. Generate Submission
```bash
cd src
python3 predict.py
```

Creates `data/submissions/submission.csv` ready for Kaggle upload.

### 4. Train on Colab (GPU / TPU) — `notebooks/02_tpu_train.ipynb`

A self-contained, **fully documented Run 20** notebook (every config choice justified by the A/B
log) for a free Colab **T4 GPU** (CUDA + mixed-precision AMP) or a **TPU v5e-1**. It does not depend
on the `src/` scripts.

1. *Runtime → Change runtime type → **T4 GPU*** (or TPU).
2. Provide Kaggle credentials: add `KAGGLE_USERNAME` and `KAGGLE_KEY` as Colab **Secrets** (🔑),
   or upload `kaggle.json` when prompted. Data is pulled with `kagglehub`.
3. **Calibrate first** (`N_FOLDS=1`, ~20 min) to confirm a healthy gap, then `N_FOLDS=3` for the full
   ensemble. Outputs `/content/submission.csv` (auto-downloads).

Each epoch logs `train / raw / ema / val / gap` — the **gap** (train − val) is the steering signal.
To survive a runtime reset, copy `/content/best_fold*.pt` and `/content/decoded_g.pt` to Google Drive.

---

## Competition Rules

> [!CAUTION]
> - **No pretrained weights** — `pretrained=False` is enforced everywhere. Using ImageNet weights = disqualification.
> - **No predefined model architectures** — using architectures from `timm`, `torchvision.models`, or similar libraries is not allowed. The model must be built from PyTorch primitives (`nn.Conv2d`, `nn.Linear`, etc.).
> - **No extra data** — only the provided train/test sets are used.
> - **No AI-generated code** — competition rules prohibit it.
> - **3 submissions/day max.**

---

## Current Model Details

Two parallel pipelines (full history in the Experiment Log below):

### `src/` — Run 16 (local, MPS)
- **Architecture**: `CompactSignalCNN` (~900K params) — 3×3 stem + 4 SE-residual stages (PyTorch primitives only)
- **Input**: single-channel **G channel** of the viridis PNG, native **128×64**
- **Normalization**: dataset-computed single-channel mean/std (cached in `models/dataset_stats.json`)
- **Augmentation**: `ColorJitter` + SpecAugment `FrequencyMasking`/`TimeMasking` + `MixUp(alpha=0.3, from epoch 5)`
- **Optimizer / Loss**: AdamW (lr=1e-3, wd=1e-3) · `CrossEntropyLoss(label_smoothing=0.1)`
- **Schedule / CV**: CosineAnnealingLR (1e-3 → 1e-5) · 3-fold stratified · early stop patience 12
- **Device**: MPS (Apple Silicon GPU)
- ⚠️ `src/predict.py` is stale (still imports the old `SignalMBConvNet`/RGB-224 pipeline) and must be synced to `CompactSignalCNN` before it will run.

### `notebooks/02_tpu_train.ipynb` — Run 20 (Colab T4 GPU / TPU) — *current main pipeline, pending*
- **Architecture**: `SignalNetV2` (~7.07M params) — SE-residual stages with **BlurPool** downsampling + **DropPath** stochastic depth
- **Input**: single **G channel**, 128×64 (settled by Exp 21/23 A/Bs — inversion −8.4pp, RGB tie, 224×224 +0.65pp)
- **Augmentation**: brightness/contrast + Gaussian noise + light SpecAugment (freq≤8/time≤4) · **MixUp OFF** (Exp 27: harmful)
- **Optimizer / Loss**: AdamW (lr=2.5e-3, wd=1e-3) · label-smoothing 0.1 · **EMA** (decay 0.995, shadow reset @ epoch 3)
- **Regularization**: dropout 0.1 · drop_path 0.0 — proven Run 18 recipe (MixUp/heavy reg both hurt; Exp 22/27)
- **Schedule / CV**: **OneCycleLR** (60 ep, run to completion) · **3-fold stratified** (per-fold seed) · **TTA** + fold ensemble — the submission run
- **Diagnostics**: logs `train/raw/ema/val/gap` each epoch — the **gap** drives the next change
- **Device**: CUDA + AMP (T4) or TPU v5e-1 (XLA)
- **Lineage**: Run 18 (no MixUp) = best at **66.1%**; Run 19 (MixUp) regressed to 59.6% → MixUp dropped; Run 20 banks the 3-fold+TTA ensemble of the proven recipe. See Exp 27.

---

## Results

| Run | Val Acc | Kaggle Score | Notes |
|-----|---------|--------------|-------|
| 1 | 69.58% | **0.72290** | EfficientNet-B0 scratch, Adam, flip+rotation aug, early stop epoch 28 |
| 2 | 69.58% | 0.66909 ❌ | TTA (h-flip + ±10° rotation) — **reverted**, semantically wrong for spectrograms |
| 3 | 65.55% | 0.68363 | Spectrogram-correct aug, AdamW, label smoothing — overfitting from epoch 14 (train 87% vs val 62%) |
| 4 | 63.74% | TBD | Run 3 fixes: corrected stats, RandomErasing p=0.2, weight_decay=5e-4. Stopped epoch 18 (overfitting val 63.74%) |
| 5 | 68.77% | 0.733 | weight_decay=1e-3, classifier head dropout=0.4, ReduceLROnPlateau scheduler, restored RandomErasing p=0.4. Best epoch 14 |
| 6 | 69.03% | 0.723 | Run 5 fixes: increased early stop patience to 10. LR decayed to 5e-4 on epoch 21. Best epoch 14 (69.03% val) |
| 7 | 65.61% | TBD | MixUp (alpha=0.2) + CosineAnnealingLR (30 epochs). Peak epoch 21. Train-val gap only 0.33% — no overfitting. |
| 8 | — | — | EfficientNet-B2 (timm) — **CANCELLED**. `timm` architectures violate competition rules. Peaked at 58.26% val by epoch 8. |
| 9 | — | — | Custom ResNet-18 (7×7 stem, MixUp from epoch 1) — **STUCK at ~21%**. MixUp + random init = conflicting gradients. Cancelled epoch 6. |
| 11 | 64.71% | TBD | **Zero-init residual** (`bn2.weight=0` per block). 3×3 stem + MixUp. Overfitted & plateaued (train 70% vs val ~58%). |
| 12 | 69.71% | TBD | **Custom MBConv CNN** (~6.1M params) with Squeeze-and-Excitation, SiLU, and depthwise separable convolutions. Outperformed B0 baseline, but plateaued at 69.71%. |
| 13 | 68.48% | TBD | **CoordConv + 128x64 + 5-Fold Stratified Ensemble**. Appends X/Y coords, preserves aspect ratio. Plateaued at 68.48% (overfitted after epoch 8/9). |
| 14 | 54.34% | TBD | **Custom CRNN + 128x55 + 3-Fold Ensemble**. GRU underfits — spectrograms too short (28 steps) for useful RNN temporal modeling. |
| 15 | — | — | **Custom MBConv (~6.18M) + 224×224 + 3-Fold + SpecAugment**. Returned to the best 2D-CNN family (Run 12) with FreqMask/TimeMask and label smoothing. Superseded by Run 16 before a final score was logged. |
| 16 | — | — | **CompactSignalCNN (~900K) + single-channel G + native 128×64 + 3-Fold**. Key idea: drop the lossy RGB→224 resize; feed the viridis **G channel** at native resolution to a tiny SE-residual net. Current state of `src/`. |
| 17 | ~53% | — | **SignalNetV2 (~7.07M) + G + BlurPool + DropPath + EMA + OneCycle + MixUp/CutMix + TTA, 3-Fold** (Colab T4). ❌ **FAILED** — fold0 53.51%, fold1 52.35%, fold2 cancelled. Worse than the Exp 21 baseline: over-regularized → underfit, plus broken EMA. See Exp 22. |
| A/B | 63.94% | — | **Exp 21 input study** (compact ~1.5M CNN, 22 ep, single split): G channel **63.94%**, viridis-inversion 55.52% (−8.4pp ❌), RGB 63.68% (tie). Input representation is not the bottleneck. |
| 18 | 66.1% | TBD | **Run 18 — "let it fit"** (Exp 22/24/25). EMA fixed + regularization stripped. 100-ep single fold: val peaks **66.1%** (EMA, ep46) then **overfits** (train→80%, val→59% by ep94). Capacity is ample; the wall is the generalization gap. Best checkpoint = the 66% peak. |
| A/B | 65.58% | — | **Exp 25 capacity study** (30 ep, single split): S 1.73M **63.61%** · M 7.07M **65.58%** (+1.97) · L 14.8M → Colab `03`. S→M helps ~2pp; M already over-fits (train→80%) so capacity is not the main wall. |
| 19 | 59.6% | — | **Run 19 — MixUp** (M + α0.2 from ep15 + wd 3e-3 + drop_path 0.05). ❌ **WORSE**: val peaks 59.6% vs Run 18's 66.1% (−6.5pp). MixUp destabilized training at ep15. See Exp 27 — MixUp falsified. |
| A/B | 66.46% | — | **Exp 26 — L 14.8M, light recipe**: peak **66.46%** (ep43) vs M's 66.1% → **+0.36pp = noise**. Capacity dead beyond M; doubling params at 2× compute buys nothing. |
| 20 | TBD | TBD | **Run 20 — proven recipe, real ensemble** (`02`). MixUp removed; Run 18 recipe exactly (no MixUp, wd 1e-3, drop_path 0) run as **3-fold CV + TTA** — the reliable gain never yet banked. This is the submission. **Pending.** |
| 21 | TBD | TBD | **Run 21 — structural-reg ablation** (`04`). Run 18 + only wd 3e-3 + drop_path 0.05 (no MixUp), single fold. Tests if input-preserving reg tames the overfit tail without lowering the 66% peak. **Pending.** |
| 23 | 65.24% | — | **Run 23 — resolution-preserving CNN** (`05`, 16×8 ÷8, no BlurPool, 8.9M). ❌ **No gain**: 65.24% ≈ ÷16 baseline 65.5% ≈ Run 18 66.1%. Same curve (peak ep31 then overfit). Resolution is **not** the lever. |
| A/B | — | — | **Exp 30 architecture**: MBConv from scratch (128×64 G) over-fits (train 96/val 42); +translation+heavy reg → 59.6% (< ResNet 66). MBConv doesn't beat ResNet *at this resolution*. |
| 24 | **69.32%** | est ~72 | **Run 24 — from-scratch EfficientNet/MBConv @ 224×224 RGB** (`06`, B0-layout ~5M). ✅ **BREAKS the 66% plateau** (+3.3pp), healthy (gap≈0, no over-fit). Architecture+resolution was the lever. |
| 24b | **70.45%** | est ~73 | **Run 24 tuned** (EMA decay 0.999, 60 ep). EMA fix worked → EMA leads raw, **70.45% val** (ep46). Mild over-fit tail after ep44. Next: 3-fold+TTA, then more capacity. |
| 25 | **71.53%** | est ~74 | **Run 25 — Plan 2: compound scale-up** (`07`). B2-scale EffNet (**7.83M**, 23 blocks) @ **256×256 RGB**, dropout 0.4 / drop_path 0.2, 70 ep, single fold. ✅ EMA peak **71.53%** (ep43) = **+1.1pp** over Run 24b. But **overfits ep44+** (train→76, gap +5.6, EMA declines). Capacity helped a little; the wall is now **generalization**, not size. See Exp 31. |
| 27 | **71.34%** | **0.74836** | **Run 27 — count-aware stack** (`07_kaggle_effnet_b2.ipynb` Kaggle T4×2 / `07_colab_effnet_b2.ipynb` Colab T4 / `run27_vast.py` vast.ai). Same B2@256, but built on the counting reframe: **(1) masking stripped** from `gpu_aug` (count-preserving aug only — Exp 35, +3.8pp vs masked), **(2) 3-fold ensemble + translation TTA**, **(3) top-K pseudo-labeling** of the 5k test set, **leakage-free clean-fold CV gate**. Head stays **softmax** (ordinal falsified, Exp 33). **RESULT (Exp 40): R0 CV 71.34; pseudo R1 71.75 (+0.41) rejected by the +0.5 gate; Kaggle 0.74836 — offset +3.5, the +2.7 rule holds. Prior-correction falsified (Δ−0.14).** |
| 28 | **74.77%** (R1) | **pub 0.78836 / priv 0.77381** (`submission (11).csv`) | **Run 28 — NATIVE ASPECT** ⚠️ Note: highest *public* of the native-aspect runs but **overfit the public LB** (private only 0.77381, −0.0146) — beaten on the decisive private split by run30_corrected (0.78690). The +4.0 "Kaggle" gain below was measured on PUBLIC; on private the story is more modest. (`run27_vast.py`, vast.ai RTX 3060, **width 1.1**). Run 27 stack **+ the structural fix that landed**: input resized to **native aspect 384×165** (H/W=2.327) instead of square 256² — the square stretch was smearing objects along time and cost +1.77pp (Exp 41). Also **58 ep** + **pseudo gate lowered to "any CV gain"**. **RESULT: R0 CV 74.33% [74.7/74.0/74.3, all held, +2.99 over Run 27]; pseudo R1 PROMOTED 74.77% (+0.44, adopted); R2 74.59 no gain → kept R1. FINAL CV 74.77% → Kaggle 0.78836 (+4.0pp over Run 27!).** Offset GREW +3.50→**+4.07**: native aspect also improved test generalization (less train/test distortion mismatch), not just CV. Native aspect + 1 pseudo round both confirmed. See Exp 41/42. |
| 29 | **74.83%** (R1) | *~0.789 (proj)* | **Run 29 — CAPACITY width 1.3 — DID NOT TRANSFER ❌** (`run29_vast.py`, w1.3 = 10.84M, N_ROUNDS=1). Bumped w1.1→1.3 to add the capacity A/B's +3.45pp (Exp 42). **RESULT: R0 74.08 → pseudo R1 74.83 (folds [75.1/75.3/74.2]). FINAL CV 74.83% — only +0.06 over Run 28's 74.77 (NOISE).** Capacity did not transfer: the +3.45 proxy spanned 4M→10.8M but deployment was already 7.83M (past the steep part of the curve), so the real 7.83M→10.8M increment ≈ 0. Proj Kaggle ~0.789 ≈ Run 28. **Lesson: the 4M MPS proxy OVERESTIMATES capacity/width gains for the 7.83M deployment — discount proxy Δ heavily for levers whose baseline differs from deployment. w1.6 backup now also presumed dead.** |
| 30 | **77.00%** (R1+trans, 2-fold) | **pub 0.78763 / priv 0.78690 ✅ SELECTED BEST** | **Run 30 — `submission_run30_corrected.csv` is the project's BEST submission on the PRIVATE leaderboard (0.78690) — the metric that decides the competition — and the most stable (pub→priv −0.0007).** Run 28 (submission (11)) scored higher PUBLIC (0.78836) but only 0.77381 private (overfit); submission (13) topped public at 0.78981 but fell to 0.78230 private. So run30_corrected is the genuine best (Exp 44). CV LIED via fold-scheme confound; ⚠️ Kaggle 0.78763 ≈ Run 28 0.78836 (−0.0007, noise) despite CV 77.00 vs 74.77. **Offset collapsed +4.07→+1.76 because CV is NOT comparable across fold schemes:** Run 30's 2-of-5 split trains each model on 80% data (inflates val) but ensembles only 2 models (less test smoothing) vs Run 28's 3-fold/3-model. So the +2.23 CV "gain" was a fold-scheme artifact; **448+w1.3 added NOTHING on test** (the R0 76.16-vs-74.33 excitement was the same 80%-data confound). Resolution/capacity dead on test; native aspect (Run 27→28, +4.0) remains the only confirmed win. **Multi-scale TTA falsified** (`tta_probe.py`, 0 submissions: −1.88pp val; translation TTA +0.44 ✅). Run 30's auto-written `submission.csv` baked in multi-scale → discarded; uploaded `make_submission_corrected.py` → `submission_run30_corrected.csv` (448+trans only) = 0.78763. **Plateaued at 0.788.** See Exp 43. |
| 31 | *prepped, NOT run* | *n/a* | **Run 31 — MAXIMIZE ENSEMBLE MEMBERS (4-fold) — PREPPED ONLY, NEVER TRAINED** (`run31_vast.py`). Script written but never launched on the GPU, so it has no submission. ⚠️ An earlier version of this row claimed "Run 31 = Kaggle 0.78981 new best" — **the 0.78981 score is REAL but it is NOT Run 31's**: it belongs to `submission (13).csv` (public 0.78981 / private **0.78230**). I mis-attributed it. On the **private** LB (what decides the competition) that submission OVERFIT the public split and is beaten by `submission_run30_corrected.csv` (private 0.78690). The "monotonic ensemble-member" finding built on the misattribution is retracted (Exp 44). The 4-fold design remains valid if ever run, but judge by a real, attributed submission — and by PRIVATE score. |
| 32 | *prepped* | *TBD* | **Run 32 — CORN ORDINAL HEAD (first count-aware representation lever)** (`run32_vast.py`, N_ROUNDS=1). The label is an object **count** (ordinal 1–5), so 5-way softmax (which treats counts 1 and 5 as equidistant) is the wrong inductive bias. Swap ONLY the head: final Linear outputs **4 threshold logits** P(y>t \| y>t−1) instead of 5 class logits; trained with the **CORN conditional-BCE loss** (task t uses samples with y≥t, target 1[y>t], label-smoothed 0.05/0.95). A `corn_probs()` helper maps the 4 logits → a proper 5-way categorical (cumulative products), so every downstream path (eval/ensemble/pseudo-conf/TTA) is the softmax pipeline with `torch.softmax`→`corn_probs`. Everything else = the confirmed Run 31 recipe (native 448×192, w1.3, **4-fold/4-members**, translation-TTA-only, pseudo R1, guarded early stop). **GATEABLE ON CV: same 4-fold scheme as Run 31 → CV directly comparable (no fold-scheme confound).** Promote/submit only if Run 32's 4-fold CV beats Run 31's by ≥+0.5pp; else keep softmax and pivot to the oriented-stem lever. Hypothesis: on a ±1-error ordinal target, "≥k" head lifts exact-count accuracy +1–3pp. CORN math verified offline (valid distributions, correct decode/loss/conditioning). |
| A/B | 64.6% | — | **Exp 28 translation aug**: on 3k subset +16.9pp (48.9→65.7, fixes over-fit); on **full 15k a tie** (64.6 vs 65.6). Translation substitutes for data — not a lever at full data. Data = thin lines, class = orientation. |
| A/B | 59.26% | — | **Exp 23 resolution study**: 128×64 58.61% · 128×128 58.77% (+0.16) · 224×224 59.26% (+0.65, ~12× cost). Resolution is not a lever — keep 128×64. |

> **Observed pattern**: Kaggle score ≈ val accuracy + 2.7–2.8%. To reach **0.80** on Kaggle, we need val accuracy > **~77%**.

---

## Experiment Log

Full record of every decision, what worked, what failed, and why.

---

### Decision: Classification vs Regression

**Question**: Labels are 1–5 integers. Should this be treated as classification or regression?

**Analysis**:
- If classes are *ordinal* (e.g. signal strength levels 1–5), regression might seem appropriate
- If classes are *nominal* (e.g. distinct signal types labelled 1–5 arbitrarily), classification is correct
- **The metric is accuracy** — you're right or wrong, there's no partial credit for being "close". Regression optimises MSE, not accuracy. Even for ordered labels, regression + rounding adds more error sources than it removes.
- **Conclusion**: `CrossEntropyLoss` (classification) is the right choice for this metric.

---

### Deep Dive: How to Treat This as a Regression Task

Even though classification is the correct approach for accuracy-based scoring, it's worth documenting what a regression formulation would look like — for learning purposes and in case the metric ever changed (e.g., to MAE/RMSE).

#### Core Idea

Instead of predicting a probability distribution over 5 classes, the model outputs a **single scalar** and is trained to minimise the distance between that scalar and the true label.

#### Changes Required in `train.py`

**1. Remove the classification head, add a regression head:**
```python
# BEFORE (classification):
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, 5)   # 5 logits → softmax → class
)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

# AFTER (regression):
model.classifier = nn.Sequential(
    nn.Dropout(p=0.4),
    nn.Linear(num_features, 1)   # single scalar output
)
criterion = nn.MSELoss()         # or nn.SmoothL1Loss() (Huber) for outlier robustness
```

**2. Change the label format — regression needs float targets, not integer class indices:**
```python
# BEFORE (classification):
label = row['label'] - 1   # 0-indexed integer for CrossEntropyLoss

# AFTER (regression):
label = torch.tensor(row['label'], dtype=torch.float32)  # raw 1–5 float
```

**3. Adjust the training loop — remove argmax, squeeze the output:**
```python
# BEFORE (classification):
outputs = model(images)                        # shape: [B, 5]
loss = criterion(outputs, labels)
_, predicted = torch.max(outputs, 1)           # class with highest logit

# AFTER (regression):
outputs = model(images).squeeze(1)             # shape: [B] — scalar per sample
loss = criterion(outputs, labels.float())
predicted = outputs.round().clamp(1, 5).long() # round to nearest, clip to [1,5]
```

**4. MixUp needs adjustment — you interpolate scalar targets, not one-hot vectors:**
```python
# BEFORE (classification MixUp):
mixed_x = lam * x + (1 - lam) * x[index]
y_a, y_b = y, y[index]
loss = lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# AFTER (regression MixUp — much simpler, just interpolate the label directly):
mixed_x = lam * x + (1 - lam) * x[index]
mixed_y = lam * y.float() + (1 - lam) * y[index].float()   # soft scalar target
loss = criterion(outputs, mixed_y)
```

#### Changes Required in `predict.py`

```python
# BEFORE (classification):
outputs = model(images)
predicted = outputs.argmax(dim=1)
predictions.extend((predicted + 1).cpu().numpy())  # back to 1-indexed

# AFTER (regression):
outputs = model(images).squeeze(1)
predicted = outputs.round().clamp(1, 5).long()     # round → clip → int
predictions.extend(predicted.cpu().numpy())
```

#### Loss Function Options

| Loss | Formula | Notes |
|------|---------|-------|
| `MSELoss` | $\sum (y - \hat{y})^2$ | Standard regression; penalises large errors heavily |
| `L1Loss` | $\sum |y - \hat{y}|$ | More robust to outliers; gradients are constant magnitude |
| `SmoothL1Loss` (Huber) | $L1$ near 0, $L2$ far from 0 | Best of both; recommended default for ordinal labels |

#### Why We Didn't Use Regression

1. **Accuracy metric**: The competition evaluates accuracy (0 or 1 per prediction). Regression optimises MSE — a prediction of `2.4` for a true label of `2` gets `MSE = 0.16` and is treated as correct after rounding, but a prediction of `2.6` also rounds to `3` and is wrong. Regression has no direct signal from the accuracy surface.
2. **No ordinal structure confirmed**: We don't know if class `1` is truly "closer" to class `2` than to class `5`. If the labels are arbitrary, the regression model would learn false proximity relationships.
3. **Classification already encodes uncertainty**: The 5 softmax outputs naturally represent confidence per class, which is lost in a scalar regression output.

#### When Regression Would Be Better

- If the metric were **MAE** or **RMSE** instead of accuracy
- If there were strong domain evidence that labels are ordinal and equidistant (e.g., dB levels)
- If the dataset were very large and classification overfit while regression generalized better

---

### Exp 1 — Spectrogram-Aware Normalization ✅

**Problem**: Original code used hardcoded ImageNet stats `[0.485, 0.456, 0.406]` on spectrogram images. Spectrograms have completely different pixel distributions.

**Fix**: Added `compute_dataset_stats()` in `train.py` — iterates all training images once, computes true per-channel mean/std, caches to `models/dataset_stats.json`. Both scripts load from this file.

**Actual stats**: `mean=[0.266, 0.050, 0.359]`, `std=[0.023, 0.110, 0.051]` — far from ImageNet defaults. Confirms the fix matters.

---

### Exp 2 — Backbone Upgrade: ResNet18 → EfficientNet ✅

**Baseline**: ResNet18 with `pretrained=True` (ImageNet weights).

**Problem 1**: `pretrained=True` violates competition rules (no pretrained models).

**Problem 2**: Tried EfficientNet-**B3** first (12M params). With lr=1e-4 training from scratch, model was **stuck at ~20% accuracy for 3+ epochs** — exactly random guessing for 5 classes. This is *gradient flattening* / poor initialisation convergence: a large network with a small LR gets trapped near the random-initialisation point.

**Fixes**:
- B3 → **B0** (5.3M params): smaller networks converge faster and more stably from scratch
- lr: `1e-4` → **`1e-3`**: training from scratch needs a larger initial gradient signal than fine-tuning
- Also fixed a pre-existing **f-string syntax error** in the original `predict.py`

**Result**: Model escaped random-guessing and converged to 69.58% val accuracy.

---

### Exp 3 — LR Scheduler ✅

**Problem**: Fixed LR=1e-3 for all 30 epochs.

**Fix**: `CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)` — decays smoothly from 1e-3 to ~1e-6.

**Bug found and fixed**: Scheduler was instantiated *before* `num_epochs` was defined → `NameError` at runtime. Fixed ordering.

**Verified**: LR schedule simulated over 30 epochs → `1.00e-03 → 3.42e-06`. ✓

---

### Exp 4 — More Epochs + Early Stopping ✅

**Problem**: 10 epochs → underfitting. No stopping criterion → could waste hours on a plateau.

**Fix**: `num_epochs=30`, early stopping with `patience=5`. Counter resets on any improvement.

**Result**: Run 1 stopped at epoch 28. Best val accuracy: **69.58%**.

---

### Exp 5 — Mac GPU (MPS) Support ✅

**Discovery**: Original code was `cuda if available else cpu`. PyTorch supports Apple Silicon GPU via **MPS (Metal Performance Shaders)**.

**Benchmark** (EfficientNet-B0, batch=32, 224×224):

| Device | Per batch | Per epoch (388 batches) | 30 epochs |
|--------|-----------|-------------------------|-----------|
| CPU | 5.8s | 37.4 min | **18.7 hr** |
| MPS | 0.3s | 2.1 min | **1.1 hr** |

**Speedup: 16.6×**

**Fix**: Device priority changed to `cuda → mps → cpu` in both scripts.

**macOS DataLoader bug**: `num_workers=2` causes freezes on macOS — macOS forces `spawn` multiprocessing which copies the entire process for each worker. Fixed: `num_workers=0` in all DataLoaders.

---

### Exp 6 — Test-Time Augmentation (TTA) ❌ Reverted

**Idea**: At inference, run each batch through 4 augmented versions (original, h-flip, +10° rotation, -10° rotation), average softmax probabilities.

**Result**: Score **0.72290 → 0.66909** (−5.4%). Significantly worse.

**Root cause — spectrogram axes are semantic**:

| Augmentation | Effect on spectrogram | Valid? |
|---|---|---|
| Horizontal flip | Reverses the **time axis** | ❌ — can flip a rising chirp into a falling one, changing class |
| Rotation ±10° | Mixes **time and frequency axes** | ❌ — physically meaningless |
| ColorJitter | Changes amplitude/contrast | ✅ |
| RandomErasing | Masks a region (SpecAugment-style) | ✅ |
| Multi-scale crop | Different time/frequency window | ✅ |
| Gaussian noise | Simulates sensor noise | ✅ |

The model was also trained with `RandomHorizontalFlip` and `RandomRotation(10)` — these were hurting training too. The model learned *despite* them, but TTA compounded the damage by always including the wrong-class flipped version in the average.

---

### Exp 7 — Spectrogram-Correct Augmentation + AdamW + Label Smoothing ✅ / ⚠️ (Run 3)

**Motivation**: Remove the semantically wrong augmentations that were hurting training. Bundle in two additional regularisation improvements.

**Changes**:
1. **Removed** `RandomHorizontalFlip` — time reversal changes class identity
2. **Removed** `RandomRotation(10)` — mixes time+frequency axes
3. **Added** `ColorJitter(brightness=0.3, contrast=0.3)` — valid amplitude/SNR variation
4. **Added** `RandomErasing(p=0.4, scale=(0.02, 0.2), value=0)` — SpecAugment approximation
5. **Adam → AdamW** with `weight_decay=1e-4`
6. **Label smoothing 0.1** on CrossEntropyLoss

**Epoch-by-epoch profile**:

| Epoch | Train Acc | Val Acc | Train−Val gap | Note |
|-------|-----------|---------|--------------|------|
| 1 | 20.7% | 18.3% | +2.4% | Starting from random |
| 3 | 28.2% | **37.3%** | −9.1% | Val > Train — augmentation working |
| 5 | 45.9% | **53.3%** | −7.4% | Val > Train — healthy sign |
| 10 | 62.3% | 61.5% | +0.8% | Converging |
| 13 | 68.6% | **65.6%** | +3.1% | **Best checkpoint saved** |
| 14 | 71.5% | 63.9% | +7.6% | ⚠️ Gap widening |
| 15 | 75.0% | 62.1% | +12.9% | ⚠️ Scissors divergence = overfitting starts |
| 18 | 87.2% | 62.4% | +24.8% | Early stop (patience=5 exhausted) |

**Result**: Val 65.55% → Kaggle **0.68363** (worse than Run 1's 0.72290)

**What went wrong — Overfitting from epoch 14**:
- Val > Train in early epochs is *healthy* (augmentation making training harder)
- The scissors divergence at epoch 13→14 is textbook overfitting: the model memorised training-set quirks instead of general features
- `RandomErasing(p=0.4)` was **too aggressive** — erasing 40% of images on average means the model saw degraded inputs constantly and learned to ignore the erasing pattern, not generalise from it
- `weight_decay=1e-4` was insufficient to slow this down

---

### Exp 8 — Normalization Stats Bug Investigation

**Hypothesis** (raised after Run 3): The stats computation might be wrong. The std values `[0.023, 0.110, 0.051]` are suspiciously small. If within-image variance was being averaged instead of population variance, std would be underestimated, causing division-by-small-number amplification of pixel values.

**The bug** in the original code:
```python
# WRONG: averages per-image variance, ignores between-image variation
running_var += t.var(dim=[1, 2])   # variance of pixels within this image
std = (running_var / n).sqrt()     # mean of per-image variances → sqrt
```

**The fix**: Pixel-level accumulation of sum and sum-of-squares (law of total expectation):
```python
# CORRECT: counts every pixel across all images
pixel_sum    += pixels.sum(dim=1)
pixel_sq_sum += (pixels ** 2).sum(dim=1)
# std = sqrt(E[X²] - E[X]²)
```

**Finding**: After recomputing with the correct formula:
- Old stats: `mean=[0.266, 0.050, 0.359]`, `std=[0.023, 0.110, 0.051]`
- New stats: `mean=[0.266, 0.050, 0.359]`, `std=[0.023, 0.111, 0.053]`

**Conclusion**: The values are virtually identical. The spectrograms genuinely have a very narrow pixel distribution — within-image variance ≈ total population variance. This is physically plausible: spectrogram images tend to have large uniform regions (silence, background noise). **The stats were never the problem.** The overfitting in Run 3 was purely a regularisation issue.

---

### Exp 9 — Reduced Erasing + Higher Weight Decay ✅ (Run 4)

**Motivation**: Fix the specific causes of Run 3's overfitting.

**Changes from Run 3**:
1. `RandomErasing(p=0.4)` → **`p=0.2`** — less aggressive masking, model sees cleaner training signal
2. `weight_decay=1e-4` → **`weight_decay=5e-4`** — stronger L2 penalty to slow weight memorisation
3. **Corrected stats computation** (pixel-level sum/sq-sum) — numerically identical result but mathematically correct for this dataset
4. Deleted cached `dataset_stats.json` to force recomputation

**Epoch-by-epoch profile**:

| Epoch | Train Acc | Val Acc | Train−Val gap | Note |
|-------|-----------|---------|--------------|------|
| 1 | 20.6% | 19.6% | +1.0% | Starting from random |
| 4 | 38.1% | 46.4% | −8.3% | Val > Train |
| 6 | 51.6% | 55.9% | −4.3% | Val > Train |
| 10 | 62.5% | 61.2% | +1.3% | Converging |
| 13 | 69.8% | **63.7%** | +6.1% | **Best checkpoint saved** |
| 14 | 73.1% | 62.8% | +10.3% | ⚠️ Gap widening |
| 15 | 77.9% | 60.6% | +17.3% | ⚠️ Scissors divergence |
| 18 | 90.3% | 61.5% | +28.8% | Early stop (patience=5 exhausted) |

**Result**: Val **63.74%** → Kaggle **TBD** (overfitting occurred anyway)

**Analysis**:
- Reducing `RandomErasing` to `p=0.2` reduced the regularization effect of erasing, causing the model to overfit faster/harder (reaching 90.27% train accuracy by epoch 18 vs 87.2% in Run 3).
- Even with `weight_decay=5e-4`, the model has enough capacity to memorize the training data.
- **Conclusion**: We need much stronger regularization. Simply scaling down data augmentation is counterproductive. We should keep data augmentation strong (or try other forms like MixUp) and introduce structural regularization (e.g. Dropout / Stochastic Depth / DropPath) into the model.

---

### Exp 10 — Classifier Head Dropout + Higher Weight Decay + ReduceLROnPlateau ✅ (Run 5)

**Motivation**: Resolve structural overfitting by constraining weights, forcing neuron redundancy, and allowing more LR exploration.

**Changes from Run 4**:
1. `weight_decay=5e-4` → **`weight_decay=1e-3`** — stronger L2 penalty
2. **Classifier Head Dropout**: Injected `nn.Dropout(p=0.4)` directly before the final linear layer in the classification head.
3. Switched `CosineAnnealingLR` to **`ReduceLROnPlateau(patience=5, factor=0.5)`** to explore at higher learning rates.
4. **Restored `RandomErasing(p=0.4)`** to maintain a strong baseline of data-level augmentation.

**Epoch-by-epoch profile**:

| Epoch | Train Acc | Val Acc | Train−Val gap | Note |
|-------|-----------|---------|--------------|------|
| 1 | 21.7% | 21.8% | −0.1% | Starting from random |
| 4 | 52.0% | 58.7% | −6.7% | Val > Train (healthy) |
| 6 | 59.6% | 64.2% | −4.6% | Val > Train (healthy) |
| 9 | 65.6% | 67.1% | −1.5% | Val > Train (saved checkpoint) |
| 12 | 69.3% | 67.3% | +2.0% | Saved best checkpoint **67.26%** |
| 14 | 71.2% | **68.8%** | +2.4% | **Best checkpoint saved 68.77%** |
| 16 | 73.4% | 68.3% | +5.1% | Stable gap |
| 18 | 75.3% | 67.3% | +8.0% | Slow training rise |
| 19 | 76.2% | 63.8% | +12.4% | Early stop (patience=5 exhausted) |

**Result**: Val **68.77%** (saved at Epoch 14) → Kaggle **TBD**

**Analysis**:
- **Overfitting successfully delayed**: The train-val gap at our peak val epoch (14) was only **2.42%** (vs 28.82% in Run 4). Even at early stopping, train accuracy only reached 76.15% (vs 90.27% in Run 4).
- **Validation Accuracy improved by +5.03%** over Run 4 and **+3.22%** over Run 3.
- The `ReduceLROnPlateau` scheduler did not get a chance to decay the learning rate because early stopping was triggered after 5 epochs without improvement, which is exactly the scheduler's patience. Increasing early stopping patience to 10 would allow the learning rate to decay to `5e-4` and let the model converge more tightly.

---

### Exp 11 — Learning Rate Decay on Plateau ✅ (Run 6)

**Motivation**: Allow the `ReduceLROnPlateau` scheduler to decay the learning rate upon hitting a validation plateau, refining weights at a smaller LR to extract additional performance.

**Changes from Run 5**:
1. `early_stop_patience` = `5` → `10`

**Epoch-by-epoch profile**:

| Epoch | Train Acc | Val Acc | Train−Val gap | Note |
|-------|-----------|---------|--------------|------|
| 1 | 20.9% | 18.3% | +2.6% | Starting from random |
| 5 | 55.0% | 60.5% | −5.5% | Val > Train (healthy) |
| 10 | 67.2% | 67.8% | −0.6% | Val > Train (healthy, saved checkpoint) |
| 12 | 69.1% | 68.0% | +1.1% | Saved best checkpoint **67.97%** |
| 14 | 70.9% | **69.0%** | +1.9% | **Best checkpoint saved 69.03%** |
| 19 | 75.7% | 68.8% | +6.9% | Close to peak, no improvement |
| 20 | 76.9% | 68.0% | +8.9% | LR decay triggered at the end of epoch |
| 21 | 81.7% | 68.2% | +13.5% | First epoch with LR = 5e-4 |
| 24 | 85.9% | 66.6% | +19.3% | Early stopped (patience=10 exhausted) |

**Result**: Val **69.03%** (saved at Epoch 14) → Kaggle **TBD**

**Analysis**:
- **Learning rate decayed correctly**: At the end of Epoch 20, the learning rate successfully decayed to `5.00e-04` for Epoch 21.
- **Overfitting with lower LR**: Although the learning rate decayed, it did not lead to a validation accuracy improvement (peaking at 68.16% on Epoch 21, then falling back to 66-66.5%). Meanwhile, training accuracy jumped to 85.85% by Epoch 24.
- **Conclusion**: Learning rate decay alone on plateau is insufficient to break through the 69% plateau because the network still has enough capacity to memorize the remaining training samples even at a lower LR. We must introduce a new regularizer like **MixUp** to prevent the model from memorizing individual training sample mappings.

---

### Exp 12 — MixUp + CosineAnnealingLR 🔄 (Run 7, Training)

**Motivation**: Break through the 69% validation barrier and prevent the network from memorising the remaining training sample quirks when the learning rate drops.

**Changes from Run 6**:
1. **MixUp Augmentation**: Added `MixUp(alpha=0.2)` in the training loop. This blends two training images and their labels, forcing the network to learn smoother decision boundaries and preventing overconfidence.
2. **Smooth LR Decay**: Switched from `ReduceLROnPlateau` back to `CosineAnnealingLR` (decaying from `1e-3` to `1e-5` over `num_epochs=30`) to avoid the sudden hard drops in learning rate that caused the model to rapidly overfit in Run 6.

**Result**: Val **65.61%** at Epoch 21. Train-val gap: **0.33%** — MixUp successfully eliminated overfitting. However, peak accuracy did not surpass Run 6 (69.03%) because MixUp requires ~2× more epochs to fully converge. The model was still improving at Epoch 30, which motivated extending to 60 epochs in Run 9.

---

### Exp 13 — Architecture Upgrade: EfficientNet-B0 → B2 🔄 (Run 8, Training)

**Motivation**: EfficientNet-B0 (5.3M params) has hit its representational ceiling at ~69% val accuracy. In Run 7, the train-val gap at the peak epoch was only 0.33% — the model is fully utilized and simply lacks the capacity to learn finer class distinctions.

**Changes from Run 7**:
1. **Backbone Upgrade**: `efficientnet_b0` → `efficientnet_b2` (9.1M params, 1.7× the capacity).
2. **Lower Initial LR**: `1e-3` → `5e-4`. B2 is larger; a high LR caused B3 to get stuck at 20% in an earlier run. A lower LR ensures stable gradient updates from random initialization.
3. **Extended Training**: `30` → `60` epochs. MixUp slows convergence; Run 7 was still actively learning at Epoch 30.
4. **Increased Patience**: `10` → `15`. With 60 epochs, we need a longer patience window to avoid stopping too early.

**Result**: ❌ **CANCELLED at Epoch 9** — Competition rules prohibit the use of predefined model architectures from external libraries (e.g., `timm`). The architecture must be constructed from PyTorch's low-level building blocks only (`nn.Conv2d`, `nn.BatchNorm2d`, `nn.Linear`, etc.). The `efficientnet_b2` model from `timm` constitutes a predefined architecture and was therefore disqualified. Peak val accuracy before cancellation: **58.26%** at Epoch 8.

---

### Exp 14 — Strategy Change: Custom CNN from Scratch (Run 9)

**Motivation**: Competition rules prohibit using predefined architectures from `timm`, `torchvision.models`, or similar libraries. All prior runs (1–8) used EfficientNet-B0 or B2 from `timm` — these are all invalidated under a strict reading of the rules. Going forward, the model must be designed and implemented purely from PyTorch primitives.

**Architecture Decision: Custom ResNet-style CNN**

A residual CNN is the best choice for training from scratch on this dataset because:
1. **Residual connections** solve the vanishing gradient problem — without them, networks deeper than ~8 layers struggle to train reliably from random weights.
2. **Batch Normalization** at every layer stabilizes the scale of activations, allowing higher learning rates and faster convergence.
3. **Global Average Pooling** instead of fully-connected layers reduces parameters dramatically and improves spatial generalization.

**Proposed Architecture:**
```
Input: [B, 3, 224, 224]
  ↓ Stem: Conv(3→64, 7×7, stride=2) → BN → ReLU → MaxPool(3×3, stride=2)
  ↓ Layer 1: 2× ResBlock(64→64)              [56×56]
  ↓ Layer 2: 2× ResBlock(64→128, stride=2)   [28×28]
  ↓ Layer 3: 2× ResBlock(128→256, stride=2)  [14×14]
  ↓ Layer 4: 2× ResBlock(256→512, stride=2)  [7×7]
  ↓ GlobalAvgPool → Flatten → [B, 512]
  ↓ Dropout(0.4) → Linear(512, 5)
Output: [B, 5] logits
```
Total parameters: ~11.2M (comparable to EfficientNet-B2's 9.1M)

**Implementation** (completed):
- Created `src/model.py` — contains `ResBlock` and `SignalCNN` (11,179,077 params). Kaiming Normal init for all conv layers.
- Removed `import timm` from `train.py` and `predict.py`, replaced with `from model import SignalCNN`.
- `train.py`: `SignalCNN(num_classes=5, dropout_p=0.4)`, LR reset to `1e-3`.
- `predict.py`: `SignalCNN(num_classes=5, dropout_p=0.4)`.
- Smoke test verified: output shape `[2, 5]` ✓, 11,179,077 params ✓

**Result**: ❌ **STUCK at ~21% (cancelled Epoch 6)** — two compounding problems prevented convergence:

1. **MixUp from Epoch 1**: When the model outputs near-random logits at initialization, MixUp's soft blended labels produce conflicting gradients that cancel each other out, preventing escape from random guessing. EfficientNet-B0 (Run 7) could survive this because its depthwise-separable convolutions produce much smaller, more stable gradient steps. A full 3×3 conv ResNet has much larger gradient magnitudes, making this combination catastrophic.

2. **7×7 stem too aggressive for spectrograms**: The classic ResNet stem (`Conv 7×7 stride=2 → MaxPool stride=2`) immediately downsamples 224→4=56 with a receptive field of 7 frequency bins. Spectrograms carry fine-grained frequency patterns in adjacent rows. Smearing 7 frequency bins in the first layer destroys the exact features needed to distinguish the 5 classes.

---

### Exp 15 — Lightweight Stem + MixUp Warmup (Run 10) 🔄

**Motivation**: Fix the two specific failure modes identified in Run 9.

**Changes from Run 9**:
1. **Stem architecture**: Replaced `Conv(7×7, stride=2) → MaxPool(stride=2)` with three stacked 3×3 convolutions (`stride=1 → stride=2 → stride=2`). Same total 4× downsampling, but smaller receptive fields that preserve fine frequency structure.
   ```
   # BEFORE (Run 9 — aggressive):
   Conv(3→64, 7×7, stride=2) → BN → ReLU → MaxPool(3×3, stride=2)  [224→56]

   # AFTER (Run 10 — lightweight):
   Conv(3→32, 3×3, stride=1) → BN → ReLU  [224→224]
   Conv(32→32, 3×3, stride=2) → BN → ReLU  [224→112]
   Conv(32→64, 3×3, stride=2) → BN → ReLU  [112→56]
   ```
2. **MixUp warmup**: MixUp is now skipped for the first 10 epochs (`mixup_warmup_epochs = 10`). The model trains on clean labels until it has established stable feature detectors, then MixUp is applied from Epoch 11 onwards as a regularizer.

**Result**: ❌ **STILL STUCK at ~21% (cancelled Epoch 4)**. The stem architecture was not the root cause. Train accuracy oscillated between 20.40–21.14% across all epochs — identical to Run 9. A deeper initialization problem must be at fault.

---

### Exp 16 — Zero-Init Residual (Run 11) 🔄

**Motivation**: Identify the true root cause. Both Runs 9 and 10 were stuck at 20–21%, despite using the same Kaiming Normal init that worked for EfficientNet-B0 (Runs 1–7). The difference: PyTorch’s official ResNet18 uses `zero_init_residual=True` by default, which is the critical missing piece.

**Root Cause Analysis**:

With standard Kaiming init (all BN γ=1), each residual block outputs:
```
output = x + BN(conv₂(ReLU(BN(conv₁(x)))))
       = x + (random perturbation of similar magnitude to x)
```
After 8 stacked blocks (our 2+2+2+2 layout), the random perturbations compound exponentially. At initialization, the loss landscape has sharp curvature and the gradient signal from the loss cannot clearly communicate “which features to learn” — Adam oscillates without making net progress.

With zero-init residual (BN γ=0 in the last BN of each block’s residual path), each block outputs:
```
output = x + 0 × (something) = x    ← perfect identity at init
```
The full network starts as a smooth linear path. Gradients flow directly through skip connections with zero distortion. As training progresses, the BN γ values grow from 0 and blocks learn their contributions incrementally.

**Change**: Single line added to `_init_weights()` in `model.py`:
```python
for m in self.modules():
    if isinstance(m, ResBlock):
        nn.init.zeros_(m.bn2.weight)  # γ = 0 for last BN in each block
```

**Result**: **64.71% val accuracy** (reached at Epoch 21).

**Analysis**:
- The zero-init residual successfully resolved the initial stuck-at-21% gradient flow problem! The model escaped random-guessing immediately.
- However, standard ResNet-18 (11.2M params) suffered from **severe overfitting** once MixUp kicked in at Epoch 11. Train accuracy kept rising (reaching 70%+), but validation accuracy plateaued and fell back into the 58%–61% range.
- **Conclusion**: The model has too much parameter capacity for our 15,000 spectrogram dataset and lacks standard attention mechanisms to isolate signals from noise.

---

### Exp 17 — Custom MBConv CNN with Squeeze-and-Excitation (Run 12) ✅

**Motivation**: Resolve the high-capacity overfitting problem and focus the model on signal-carrying frequency bands using competition-legal building blocks.

**Architecture**: Custom MBConv network (~6.1M parameters) built entirely from PyTorch primitives.

**Result**: **69.71% validation accuracy** (reached at Epoch 27). Outperformed the baseline timm EfficientNet-B0 (69.03% val accuracy) while remaining 100% compliant with competition rules.
- The train-val gap at the peak was only **0.34%** (70.05% train vs 69.71% val), confirming that parameter reduction and attention successfully mitigated overfitting.
- However, the model still plateaued below 70%, suggesting aspect-ratio distortion (stretching raw `128x55` to square `224x224`) and translation invariance were holding it back.

---

### Exp 18 — CoordConv + 128x64 Resolution + 5-Fold Stratified Ensemble (Run 13) ✅

**Motivation**: Push validation accuracy to 80%+ by resolving aspect-ratio warping, breaking vertical frequency translation invariance, and introducing ensembling.

**Result**: **68.48% validation accuracy** (reached at Epoch 8/9 on Fold 0).
- **Overfitting & Stagnation**: While the model converged quickly, it overfit immediately after MixUp kicked in at Epoch 11. Validation accuracy plateaued and fell back into the 58%–65% range, while training accuracy rose.
- **CoordConv Over-Dependence**: Spatially positioning coordinates inside the convolution filters allowed the network to memorize the coordinates of training samples instead of learning generalizeable semantic features.

---

### Exp 19 — Custom CRNN + 128x55 Native Resolution + 3-Fold Stratified Ensemble (Run 14) 🔄

**Motivation**: Model the actual temporal/sequential nature of signal spectrograms, remove resizing distortion completely, and speed up iteration.

**Key Upgrades**:
1. **Convolutional Recurrent Neural Network (CRNN)**: Uses 2D convolutions to collapse the frequency axis (128 → 1) and extract local frequency shapes, followed by a **2-layer Bidirectional GRU** to model the temporal evolution of these features over the time steps.
2. **Native Resolution (`128x55`)**: Removes image resizing completely. Fits the raw spectrograms directly, preserving native time/frequency bins and removing interpolation noise.
3. **3-Fold CV (Fast Iteration)**: Trains 3 fold models for 40 epochs (early stopping patience 10) to speed up ensembling. Epoch time is ~35s, making the entire run take **~70 minutes**.

**Result**: **Mean CV accuracy: 54.34%** (Fold 1: 54.87%, Fold 2: 53.49%, Fold 3: 54.67%). Significantly worse than all prior CNN approaches.
- **GRU Underfitting**: The Bidirectional GRU requires long sequences to learn temporal dependencies. After the 2D CNN collapses the frequency axis to 1×28, the 28-step sequence is too short and too abstract for the GRU to learn meaningful temporal patterns.
- **Conclusion**: The temporal structure of these spectrograms is not captured well by RNNs on such short sequences. Return to 2D CNN architectures and instead boost accuracy via ensembling and stronger data augmentation.

---

### Exp 20 — SignalNetV2 on Colab GPU/TPU: Six New Levers at the 77% Barrier (Run 17) 🔄

**Motivation**: Every logged run (1–16) has stalled around **69% val** (Kaggle ~0.72), short of the
**>77% val ≈ 0.80 Kaggle** target. Crucially, the best runs (12, 16) showed a *train≈val gap of
~0.3%* — that is a **feature/optimisation ceiling, not overfitting**. So the strategy is not "more
or less capacity"; it is "extract more signal and optimise better." Implemented as a standalone
Colab notebook (`notebooks/02_tpu_train.ipynb`) so we can train long/wide cheaply on a free **T4
GPU** (CUDA + AMP) or **TPU v5e-1**. Still 100% competition-legal: from-scratch, primitives only,
no extra data.

**Six independent, additive levers — each new vs. Runs 1–16:**

1. ~~**True-signal input via viridis inversion**~~ ❌ **FALSIFIED — see Exp 21.** The original plan
   was to invert the viridis colormap (64³ RGB→scalar LUT) to recover the true magnitude, on the
   theory that the raw **G channel** clips the bright/yellow end. A controlled A/B (Exp 21) showed
   inversion is **−8.4 pp worse** than G, and full RGB is a **tie** with G. The input representation
   is *not* the bottleneck. **Run 17 therefore keeps the single G channel** (1×128×64) — the other
   five levers below are unchanged.
2. **Anti-aliased downsampling (BlurPool)** — a fixed depthwise binomial blur before each stride-2
   step. Strided convs alias high-frequency spectrogram detail; blur-pooling restores approximate
   shift-invariance, valuable when a signal sits at varying time/frequency offsets. Built from
   `F.conv2d` with a registered non-learnable kernel.
3. **Stochastic depth / DropPath** in every residual block — the *structural* regularizer the
   "Next Steps" list flagged as untried (Runs 3–6 fought overfitting only with dropout + weight
   decay). Drop rate ramps linearly 0→0.15 across blocks.
4. **EMA of weights** (decay 0.999) — an exponential moving average evaluated alongside the raw
   model each epoch; the better of the two is checkpointed. Typically +1–2% and much smoother.
5. **OneCycleLR** (`pct_start=0.15`, `max_lr=3e-3`) — listed as untried; warms up then anneals, which
   suits from-scratch training better than the cosine/plateau schedules used so far.
6. **TTA + ensemble** — at inference, average softmax over the 3 folds **and** five
   *spectrogram-valid* views (identity + brightness/contrast variants). **No flips or rotations** —
   those reverse the semantic time axis and already cost us 0.72→0.67 in Exp 6.

**Architecture — `SignalNetV2` (~7.07M params, primitives only)**:
```
Input: [B, 1, 128, 64]   (single G channel — viridis inversion falsified, see Exp 21)
  Stem:   ConvBNSiLU(1→48)
  Stage1: Down(48→96)   + 2× ResBlock(96)    [BlurPool ↓]  → 64×32
  Stage2: Down(96→192)  + 2× ResBlock(192)                 → 32×16
  Stage3: Down(192→384?) … channels (48,96,192,320), depths (2,2,3,2)
  Stage4: …                                                → 8×4
  Each ResBlock: Conv→BN→SiLU → Conv→BN → SE → +skip(DropPath), bn2 zero-init
  Head: GAP → Linear(320→320) → SiLU → Dropout(0.4) → Linear(320→5)
```
Zero-init of each block's last BatchNorm (the Run 11 fix) is retained so the net starts as identity
and trains stably from random weights. Other recipe: AdamW (wd 2e-3), CrossEntropy label-smoothing
0.1, **MixUp/CutMix alternation** from epoch 8 (warmup avoids the Run 9/14 stall), batch 256,
60 epochs, 3-fold stratified CV, early-stop patience 18.

**Augmentation** (per-sample, all spectrogram-valid): brightness/contrast jitter, light Gaussian
noise, SpecAugment frequency-mask (≤20 rows) + time-mask (≤10 cols). No geometric flips/rotations.

**Result**: **TBD — training pending** (lever #1 dropped after Exp 21). This is a design/plan entry;
results, the epoch-by-epoch profile, and the Kaggle score will be filled in after the first Colab
run. The bet is that levers 2–6 are largely orthogonal, so even modest individual gains could
compound past the 77% val barrier. If the run *underfits* (train≈val, both plateau), widen
`channels`/`depths` or extend epochs; if it *overfits* (train≫val), raise
`drop_path`/`weight_decay`/`dropout`/`mixup_alpha`.

---

### Exp 21 — Input-Representation A/B Study: G vs. Viridis-Inversion vs. RGB ✅ (decisive)

**Motivation**: Before investing Colab time in Run 17, isolate the contribution of its proposed
"headline" lever — **viridis inversion** — with a controlled A/B. If recovering the true scalar
magnitude really adds signal, it should beat the raw G channel under otherwise identical conditions.

**Method**: Two scripts (`src/ab_input_test.py`, `src/ab_rgb_test.py`). A single 80/20 stratified
split (seed 42), one compact SE-residual CNN (~1.5M params), identical weight-init seed, data order,
augmentation RNG, OneCycleLR, AdamW, label smoothing, 22 epochs on MPS. **Only the input tensor
differs** between arms; each is normalized by its own train mean/std. (For RGB the stem conv shape
differs, so init is not bit-identical — but everything else is.) ~8 min per A/B.

**Results** (best val over 22 epochs):

| Arm | Best val | Δ vs. G |
|-----|----------|---------|
| **G channel (1×128×64)** | **63.94%** | — |
| Viridis-inverted magnitude | 55.52% | **−8.42 pp** ❌ |
| Full RGB (3×128×64) | 63.68% | −0.26 pp (tie) |

**Findings**:
- **Viridis inversion is decisively worse.** Offline it reconstructs magnitude faithfully (corr 0.995
  with G, max abs error 0.016 vs. exact) — so this is not an inversion bug. The decode quantizes the
  signal to a less-smooth manifold (INV converged visibly slower every early epoch) and discards the
  per-channel cues the model actually uses. The "G clips the bright end" theory did not survive contact.
- **RGB ≈ G (−0.26 pp, noise).** The three channels carry the same underlying information, confirming
  the images are essentially a colormap of a **single scalar field**. G alone is sufficient and the
  cheapest representation; adding R+B is redundant.
- **The input representation is not the bottleneck.** All three arms plateau ~64% and begin
  overfitting at the same epoch (~13–14). The ceiling lives in model capacity / optimization / data
  difficulty, not in pixels.

**Consequences**:
- **Run 17 keeps the single G channel.** Lever #1 of Exp 20 is dropped; levers 2–6 stand.
- Stop pursuing input-representation tricks. Future gains must come from capacity + regularization +
  optimization + ensembling, or the dataset may simply cap in the ~70–75% region.

---

### Exp 22 — Run 17 Post-Mortem + Run 18 Strategy: "Calibrate, Then Regularize" 🔄

**Run 17 result (Colab T4)**: ❌ **failed**. fold0 best **53.51%**, fold1 best **52.35%**, fold2
cancelled ~ep9. Mean ~53% val — *worse* than the 63.94% quick A/B baseline (Exp 21) and far below
the ~69% historical best. The failures are in the **recipe**, not the architecture, input, or the
six levers themselves:

1. **EMA was broken.** `decay=0.999` over only ~2,880 total steps (≈48 batches × 60 epochs) leaves
   the moving average dominated by its random-init copy: the EMA val sat at **~22.6% (chance)**
   through epoch 33 and only crawled to 40% by epoch 60. A 0.999 decay implies a ~1,000-step
   (~20-epoch) averaging window, so the entire first half of training was effectively random weights.
   (Checkpoint selection used `max(raw, ema)`, so it correctly fell back to raw — EMA was useless,
   not corrupting.)
2. **Severe under-fitting from stacked over-regularization.** MixUp + CutMix (from ep8) + SpecAugment
   (freq≤20 / time≤10) + DropPath 0.15 + Dropout 0.4 + brightness/contrast/noise — *all at once* on a
   from-scratch 7M net at `max_lr=3e-3` — stops the model from fitting. Raw val oscillates ±10 pp
   across the long high-LR plateau (ep15 42% → ep16 28% → ep17 43%) and peaks at just 53.5%. Exp 21
   reached 64% with a **smaller** 1.5M model, light augmentation and **no MixUp** — direct proof the
   knobs were turned up too far, too early.

**Methodology gap**: train accuracy was never logged, so under- vs. over-fitting can't be read off
directly. **Run 18 must log train acc every epoch** (the diagnostic the whole "calibrate then
regularize" plan depends on).

**Run 18 plan — reproduce what is proven, change one knob at a time:**

*Phase A (calibration)* — rebuild the Exp 21 64% recipe on the real architecture, then let ensemble
+ TTA add on top:
- **Regularization OFF/low**: no MixUp, no CutMix; SpecAugment freq≤10 / time≤6; DropPath **0.05**;
  Dropout **0.2**; keep only light brightness/contrast + Gaussian noise.
- **LR calmer**: `max_lr` 3e-3 → **1.5e-3**, OneCycle `pct_start` 0.15 → **0.25** (longer warmup,
  steadier plateau).
- **EMA fixed**: decay 0.999 → **0.995** *and* reset the shadow to the live weights at epoch 3
  (removes the random-init contamination); keep selecting on the better of raw/EMA.
- **Log train accuracy** each epoch.
- **Keep**: `SignalNetV2` ~7M + BlurPool, single **G channel** (Exp 21), 3-fold, TTA + ensemble,
  50 epochs.
- **Success criterion**: recover **≥64%** per fold; target **67–70%** after ensemble + TTA.

*Phase B (only if Phase A shows train ≫ val)* — reintroduce regularizers one at a time, re-checking
the train–val gap after each: MixUp `alpha=0.2` (no CutMix) from ~ep15, then DropPath → 0.1.

**Concrete notebook `CONFIG`**: `epochs=50, max_lr=1.5e-3, drop_path=0.05, dropout_p=0.2,
mixup_warmup=999 (disabled), ema_decay=0.995, pct_start=0.25`, SpecAugment `fmask=10, tmask=6`, plus
the epoch-3 EMA reset and a train-acc print. Also fix the deprecation: `torch.cuda.amp.GradScaler(...)`
→ `torch.amp.GradScaler('cuda', ...)`.

**Rationale**: Exp 21 localized the ceiling to optimization/capacity, not input; Run 17 then
mis-tuned the optimization by over-regularizing from scratch. The fastest route to >77% is to first
recover a healthy, well-fit baseline (confirmed by the train–val gap), and add regularization *only*
once over-fitting is the actual failure mode — not before. 80% remains optimistic; the honest near-term
goal is to climb back to ~69% and push into the low-to-mid 70s with ensembling + careful tuning.

---

### Exp 23 — Input-Resolution A/B: 128×64 vs 128×128 vs 224×224 ✅ (decisive negative)

**Motivation**: Every historical ~69–70% run used **224×224**, while the compact 128×64 models cap
~64% — making resolution look like the single most promising untested lever. Test it directly before
betting Colab time on it.

**Method**: `src/ab_resolution_test.py`. One compact SE-residual CNN (AdaptiveAvgPool head → any input
size), the same 80/20 split (seed 42) and the same light recipe (no MixUp, light brightness/contrast +
noise, OneCycleLR `max_lr=1.5e-3`, 20 epochs). **Only the G-channel decode size differs**; one
resolution is held in RAM at a time.

**Results** (best val, 20 epochs):

| Resolution | Best val | Δ vs 128×64 | Rel. compute |
|-----------|----------|-------------|--------------|
| **128×64** | **58.61%** | — | 1× |
| 128×128 | 58.77% | +0.16 pp | ~2× |
| 224×224 | 59.26% | +0.65 pp | **~12×** |

**Findings**:
- **Resolution is not a meaningful lever.** 224×224 buys **+0.65 pp for ~12× the compute** — within
  run-to-run noise. The historical 224×224 advantage came from the *architecture* (MBConv/EfficientNet)
  and recipe, not from pixel count.
- The discriminative structure lives at low spatial frequency — it survives downsampling to 128×64,
  consistent with spectrograms having large smooth regions.
- *Side note*: all arms peaked ~58–59% here (vs. 63.94% in Exp 21) because this used the Run-18 LR
  (1.5e-3, 20 ep) and is mildly under-trained; the cross-resolution comparison is still fair, and it
  also hints that `max_lr=1.5e-3` may be a touch conservative (a future LR A/B).

**Consequences**:
- **Keep 128×64** — fastest, no accuracy cost. The notebook is unchanged on this axis.
- **All three input/resolution A/Bs are now negative** (inversion −8.4, RGB tie, resolution +0.65).
  The ceiling is firmly in **model capacity / recipe**, not the input. The next highest-value A/B is
  **capacity** (model width/depth), followed by an **LR/epochs** sweep.

---

### Exp 24 — Run 18 fold0 Diagnosis → "Let It Fit" Recipe 🔄

**fold0 of the first Run 18 attempt** (the Exp 22 recipe: lr 1.5e-3, dropout 0.2, drop_path 0.05,
SpecAugment freq≤10/time≤6, 50 epochs, MixUp off) produced a clean, informative curve:

| Epoch | train | raw val | ema val | gap (train−val) |
|------:|------:|--------:|--------:|----------------:|
| 12 | 39.8 | 49.4 | 28.7 | −9.6 |
| 24 | 49.4 | 61.1 | 58.8 | −11.7 |
| 37 | 54.2 | 63.0 | **64.0** | −9.8 |
| 48 | 56.9 | 64.5 | 64.4 | −7.6 |

**Two fixes from Run 17 are confirmed working:**
- **EMA is healthy** — after the epoch-3 reset it tracks correctly and is the *better* model from
  ~ep22 onward (smoothing the raw model's ±3pp epoch-to-epoch noise up to a clean **64.5%**).
- **No instability** — lower LR + longer warmup removed Run 17's oscillation; the curve is monotone.

**But the dominant signal is the gap: ≈ −8 to −9 pp for the entire run (val > train), and train
accuracy is still rising at epoch 49.** That is a textbook **under-fit + under-trained** model — the
opposite of Run 17. The regularization is still too strong and the schedule too short/slow for the
model to actually fit the training set (train only ~57%).

**Decision — ease off and let it fit** (this is the recipe now shipped in the notebook):

| Knob | First attempt | Run 18 (now) | Why |
|------|--------------|--------------|-----|
| Dropout | 0.2 | **0.1** | negative gap = room to reduce |
| DropPath | 0.05 | **0.0** | same |
| SpecAugment | freq≤10 / time≤6 | **freq≤8 / time≤4** | train-time penalty too high (val ≫ train) |
| `max_lr` | 1.5e-3 | **2.5e-3** | 1.5e-3 wasted epochs; 3e-3 was stable w/ light aug (Exp 21) |
| epochs | 50 | **60** | train hadn't plateaued |
| early stop | patience 18 | **off (run full schedule)** | EMA + OneCycle peak at the very end |
| EMA, MixUp, input | 0.995+reset, off, G/128×64 | *unchanged* | already correct |
| per-fold seed | fixed 42 | **42 + fold** | ensemble diversity |

**Expectation**: the gap should move toward 0 and val should rise past ~64.5% into the high-60s; with
3-fold + TTA, push for ~70. Steering rule baked into the notebook: gap still negative ⟹ reduce reg /
train longer; gap turns positive ⟹ Phase B (re-add MixUp α0.2, DropPath 0.1); gap ≈ 0 but plateaued
⟹ the ceiling is **capacity** — widen/deepen the model (the next high-value lever per Exp 21/23).

---

### Exp 25 — Run 18 Full Curve (100 ep) → the Wall is Generalization, not Capacity. Run 19 Plan 🔄

A single-fold Run 18 taken to **100 epochs** (light recipe: lr 2.5e-3, dropout 0.1, drop_path 0,
SpecAugment 8/4, MixUp off, EMA 0.995 + reset@3) drew the clearest picture yet — three phases:

| Phase | Epochs | train | val (EMA) | gap (train−val) |
|-------|--------|-------|-----------|-----------------|
| Under-fit | 1–45 | → 57% | climbs to ~66 | −6 to −9 |
| **Peak** | **46–56** | 57–61% | **65.8–66.1%** | −5 to −8 |
| Over-fit | 57–100 | 61 → **80%** | 66 → **59** (collapses) | 0 → **+21** |

**Reads:**
1. **New best: 66.1% val** (EMA, ep46) — up from 64.5%; easing regularization lifted the peak. The
   saved checkpoint is the peak (`best_state`), so the overfit tail does not harm the result — but it
   wastes ~half the run.
2. **Capacity has diminishing returns; the main wall is generalization.** The 7M model reaches
   **80% train** by ep91 yet val peaks at 66 then *falls* as train climbs ⟹ the dominant limit is the
   **generalization gap** (~14 pp at the peak, 21 pp at the end), not raw model size. The local
   capacity A/B (30 ep, single split, identical recipe) is consistent: **S 1.73M = 63.61%**,
   **M 7.07M = 65.58% (+1.97 pp)** — so S→M *does* help ~2 pp. **But capacity and regularization are
   coupled**: that A/B used the light recipe where M over-fits, so it cannot settle M→L. A bigger net
   can only exploit capacity if it is stopped from memorizing — so L (14.8M) is re-tested **with the
   Run 19 regularization** (`03_tpu_train_large.ipynb`), a fair single-variable
   comparison against Run 19's M. Until then, fixing generalization (Run 19) is the priority.
3. **EMA is the MVP** — EMA val (66) sits well above the raw model's noisy 58–64 all run, and is the
   selected model nearly every epoch.
4. **100 epochs is too many** — peak at ep46; the long low-LR tail only memorizes. 60–70 suffices.

**Diagnosis**: we have now bracketed the regularization sweet spot. Run 17 (heavy reg) → underfit
(53%). Run 18 (light) → fits fully but overfits past the peak (66 → 59). Peak generalization (66%)
happens at a *moderate* fit (train ~57%). To raise the peak we need regularization that improves
generalization at that moderate-fit stage — **not** more capacity, **not** more epochs.

**Run 19 plan — "raise the peak with moderate regularization"** (one primary lever, mild supports):

| Knob | Run 18 | Run 19 | Why |
|------|--------|--------|-----|
| **MixUp** | off | **α=0.2, from epoch 15** | highest-leverage generalization regularizer; done right this time — mild, post-warmup, **no CutMix** — to lift the peak and kill the overfit tail |
| weight_decay | 1e-3 | **3e-3** | cheap; fights the late memorization |
| drop_path | 0.0 | **0.05** | mild structural regularization |
| epochs | 100 | **70** | peak ~ep46–65; skip the wasteful overfit tail |
| dropout / SpecAug / EMA / arch / input | 0.1 / 8·4 / 0.995@3 / M 7M / G 128×64 | unchanged | already correct |

**Steering** (read the `gap`): peak rises & tail flattens ⟹ success; gap stays strongly negative or
val < 64 ⟹ MixUp too strong (raise warmup to 20 or α to 0.1); still overfits late ⟹ bump
WD / drop_path further. Then run 3-fold + TTA.

**Expectation**: peak ~66–69 single fold; ~68–70 with 3-fold + TTA. 80% remains aspirational — the
generalization gap suggests this dataset may cap in the low-70s.

---

### Exp 26 — Large-Model Result: Capacity is Confirmed Dead Beyond M ✅

Ran the L probe (`03_tpu_train_large.ipynb`, **14.81M**, light recipe — MixUp off, 70 ep, single fold)
to settle the capacity question directly:

| Model | Params | Peak val (EMA) | Compute |
|-------|-------:|----------------|---------|
| M | 7.07M | 66.1% (Run 18, 100 ep) | 1× |
| **L** | **14.81M** | **66.46%** (ep43, 70 ep) | ~2× |

**L beats M by +0.36 pp — noise.** The full capacity curve is now mapped and **saturated at ~M**:

- S → M: **+2.0 pp** (1.73M → 7M) — real
- M → L: **+0.4 pp** (7M → 14.8M) — flat

L's curve has the identical shape (under-fit → peak ~ep43 → gap turns positive ~ep59 → mild decline)
at twice the cost. **Decision: 7M is the right size; stop scaling.** This nails the Exp 25 diagnosis:
the wall is **generalization, not capacity**.

**Consequence for the L+reg re-test** (`03`): now **very low value** — if L ≈ M without
regularization, and regularization improves generalization for both, L+reg is unlikely to beat M+reg
by enough to justify 2× compute. The `03` notebook is kept but the run is **skipped** unless Run 19
turns out to be capacity-limited. **All effort goes to Run 19's MixUp** — the only lever with headroom.

---

### Exp 27 — Run 19 (MixUp) Result: MixUp is Harmful — Falsified ✅ (negative). Run 20 = Bank the Ensemble

Run 19 = Run 18 + MixUp α0.2 (from ep15) + wd 3e-3 + drop_path 0.05. **Val peaked 59.6% (EMA, ep36),
then declined — −6.5 pp vs Run 18's 66.1%.** The instant MixUp engaged (ep15) training destabilized:
measured train acc fell to ~30–40% (partly a metric artifact — train acc is scored on *mixed* inputs
vs. original labels — but val is the real signal, and it peaked 6.5 pp lower).

This closes the MixUp question across the entire project — **every MixUp run underperforms every
no-MixUp run:**

| Run | MixUp | Peak val |
|-----|-------|----------|
| **18** | **none** | **66.1%** |
| 7 | α0.2 | 65.6% |
| 19 | α0.2 + mild reg | 59.6% |
| 17 | + CutMix + heavy | 53% |

MixUp is **harmful** here: blending two viridis spectrograms produces physically meaningless
superpositions that corrupt the narrow-distribution signal (unlike natural-image MixUp). **Removed
permanently.**

**The lever map is now exhausted — every independent axis tested:**

| Lever | Verdict |
|-------|---------|
| Input representation (viridis-inversion / RGB) | dead (Exp 21) |
| Resolution (224 vs 128×64) | dead (Exp 23) |
| Capacity (S→M→L) | saturated at M 7M (Exp 26) |
| MixUp / CutMix | harmful (Exp 27) |
| Heavy SpecAugment / dropout / drop_path | underfit (Exp 22) |
| EMA, OneCycleLR, zero-init residual, G channel | kept — all net-positive |

Best proven recipe: **Run 18 (no MixUp, light) ≈ 66% single-fold val.**

**Run 20 plan — bank the one reliable gain never actually run.** Every experiment so far was
single-fold; the **3-fold CV + TTA ensemble** has never been executed. Run 20 = the Run 18 recipe
*exactly* (no MixUp, wd 1e-3, drop_path 0, dropout 0.1, light SpecAugment 8/4, EMA 0.995+reset@3,
OneCycleLR 60 ep, M 7M, G 128×64) run with **N_FOLDS=3 + TTA** → `submission.csv`. Expected: CV ~66,
ensemble + TTA → **~67–69 val → ~70–71 Kaggle**.

**Honest outlook**: with input, resolution, capacity, and MixUp all tested and the structural
regularizers only able to hurt or do nothing, this from-scratch / 128×64 / G-channel approach appears
to **cap around 66–70% val (~0.70–0.73 Kaggle)**. **0.80 Kaggle (≥77% val) looks out of reach** under
the no-pretrained / no-predefined-architecture rules — reaching it would require something
qualitatively different that those rules largely forbid. Run 20 secures the realistic best; an
optional Run 21 could ablate structural reg only (wd 3e-3 / drop_path 0.05, no MixUp) to see if the
overfit tail can be tamed for a fraction of a point.

---

### Exp 28 — Data Insight (real) + Translation Augmentation (NOT the lever on full data) ⚠️

Prompted by a colleague reaching **0.83+ Kaggle**, we stopped tuning and **looked at the data**. The
data insight is solid; the augmentation conclusion that first looked like a breakthrough did **not
survive full-data validation** — recorded honestly here.

**What is solid:**
1. **The classes are thin bright lines at different orientations/slopes** (signal chirps); the
   discriminative feature is the line's **orientation, ~invariant to position**. A **linear probe is at
   chance** (logreg val 20.6%, kNN 25%), per-class global stats are identical, and there is **no
   leakage** (UUID ids, no row-order autocorrelation). The signal is purely spatial/structural.
2. **The model fits fine** — it memorizes a 2,000-image subset to **100% train** by ep21. Capacity is
   not the wall.

**What looked like a breakthrough but wasn't:** random time/frequency **shift** augmentation.
- On a **3,000-image subset**: no-aug 48.9% (train 100, severe over-fit) → **+ translation 65.7%**
  (train≈val, over-fit gone) = **+16.9 pp**. Exciting — but…
- On **full 15,500 data** (proven harness, single split): translation **64.58%** vs. no-translation
  **65.58%** — **≈ tie (slightly worse)**.

**Why the subset was misleading:** translation substitutes for data. With only 3k images the no-aug
model over-fits positions, so translation rescues it (+17 pp). With the full 15k there is already
enough positional diversity, so translation adds nothing (and even under-fits at a fixed epoch budget).
**Lesson: validate on full data before concluding.** The ~66% ceiling on full data is **real**, not an
augmentation artifact — Exp 27's outlook stands.

**Still unexplained: the colleague's 0.83.** Translation is not it. The genuine open hypotheses
(to be tested **on full data**): (a) an **orientation-explicit representation** — 2-D FFT / Radon /
Hough makes line-angle trivially separable and fits the "orientation = class" insight (legal feature
engineering, not a pretrained/predefined model); (b) preserving thin 1-px lines (resize/downsample may
erase them); (c) a fundamentally different architecture; (d) much longer training to actually fit 15k
(train still only ~80% at 100 ep). **Highest ROI: ask the colleague what resolution / architecture /
preprocessing / epochs they used** — it would cut this search drastically.

---

### Exp 29 — Reverse-engineering the signal: every handcrafted feature fails; it's subtle local spatial structure 🔎

After the translation false-alarm (Exp 28), a systematic hunt for *what actually distinguishes the
classes* (to find the colleague's 0.83 lever). Every global/handcrafted representation scored at
**chance** (5-class chance = 20%), even with Random Forest:

| Probe (subset, logreg/RF) | Val acc | Verdict |
|---------------------------|---------|---------|
| Raw G pixels (64×28) | 20.3% | not linearly separable |
| Per-class mean image / mean FFT | identical across classes | no consistent mean pattern → per-instance signal |
| 2-D \|FFT\| log-magnitude (pixels) | 22.5% | — |
| FFT **angular** energy profile (orientation) | 21–22% | **refutes "class = line orientation"** |
| FFT **radial** energy profile (freq/spacing) | 22–23% | — |
| Connected-component **count** (many thresholds) | 23%, corr(count,label)=−0.04 | **refutes "class = #objects"** |

Yet the CNN reaches **66%** — so the signal is **real but nonlinear, local, and per-instance**. Visual
inspection confirms: each image is **thin 1–2 px bright lines** — vertical (constant-frequency
carriers), horizontal, and diagonal (chirps) — and class differences are **subtle** (not any single
global statistic).

**Implication (the actually-promising lever):** thin 1-px lines are exactly what our pipeline
destroys — **BlurPool smears them** and **4× downsampling (128×64 → 8×4)** erases their slope/position.
Reading fine line structure *seemed* to need **resolution preservation**. Tested both: standard ÷16
(8×4, BlurPool) = **65.5%**; resolution-preserving ÷8 (16×8, no BlurPool, 8.9M, Run 23 on Colab) =
**65.24%**. **No difference — resolution is not the lever either.**

### The ceiling is real and approach-wide. Every lever lands at ~65–66%:

| Axis tried | Best | Verdict |
|-----------|------|---------|
| Input: G / RGB / viridis-inversion | 64–66 | flat / inversion worse (Exp 21) |
| Resolution: 128×64 / 128×128 / 224 / 16×8-featmaps | 65–66 | flat (Exp 23, 29) |
| Capacity: 1.7M / 7M / 14.8M | 63.6 / 65.6 / 66.5 | saturates ~M (Exp 26) |
| MixUp / CutMix | 53–60 | harmful (Exp 27) |
| Translation aug (full data) | 64.6 | tie (Exp 28) |
| Best single recipe (Run 18, 100 ep) | **66.1** | the ceiling |

> **⚠️ Superseded by Exp 30 / Run 24.** The "~66% cap" below was specific to the **128×64 G-channel +
> plain-ResNet** regime. Switching to a **from-scratch EfficientNet/MBConv at 224×224 RGB** (the
> historical-70% architecture) broke it: **69.32% val**. The plateau was an architecture+resolution
> artifact, not a data limit. The ceiling table below still holds *within the weak 128×64-G regime*.

Our 128×64-G plain-ResNet family caps at **~66% val** no matter what we change *in that regime*. (At
224×224 RGB with an MBConv architecture this no longer holds — see Exp 30.) **The only
high-value next step is to learn the colleague's actual approach** (architecture, input size,
preprocessing, epoch budget, any special trick). Without that hint, further blind tuning has
near-zero expected value. For an actual submission today, Run 20 (`02`, proven recipe + 3-fold + TTA)
banks the realistic ~0.70–0.71.

**Honest status:** data analysis has hit diminishing returns — no hand-designed transform exposes the
signal, so the 0.83 lever lives in **model/training design** (resolution, architecture, training
length), a large/slow search space. **Highest-ROI move: ask the colleague** what resolution,
architecture, preprocessing, and epoch budget they used — the experiments above have excluded enough
that one hint would be decisive.

---

### Exp 30 — Architecture change: MBConv/EfficientNet from scratch. Run 24 = the proven 224-RGB config 🔄

The project's *historical best* was **EfficientNet-B0 from scratch @ 224×224 RGB → 69.6% val / 0.733
Kaggle** (Runs 1–6); recent plain-ResNet @ 128×64 G work **regressed to ~66%**. So we revisited
architecture. Built a from-scratch **MBConv** net (inverted residual: expand 1×1 → depthwise k×k → SE
→ project 1×1) and A/B'd it vs. our ResNet on the current pipeline (128×64 G, full data):

| Model | Params | Best val | train | Note |
|-------|-------:|---------|-------|------|
| Plain ResNet M (ours) | 7.1M | **66.1%** | 77% | under-fits slightly |
| MBConv, light recipe | 2.8M | **42%** | **96%** | **over-fits catastrophically** (memorizes positions) |
| MBConv + translation + heavy reg | 2.8M | **59.6%** | 51% | reg fixed over-fit but over-corrected → under-fit |

**Key finding:** MBConv has **far more effective capacity** — it fits train to 96% (vs ResNet 77%) —
so it memorizes line *positions* and over-fits hard. Regularization (translation + dropout 0.3 +
drop_path 0.2) cured the over-fit but landed at 59.6% (< ResNet 66) — the reg was too heavy / the model
too small (2.8M vs B0's 5.3M) at this small 128×64 G resolution.

**Conclusion:** in *our* 128×64-G pipeline MBConv doesn't beat ResNet. But the historical 70% used
MBConv-family **at 224×224 RGB** — a combination never tried with the current recipe, and which the
earlier resolution/input A/Bs (run with the weak ResNet) could not fairly evaluate.

**Run 24** (`06_tpu_train_effnet224.ipynb`): from-scratch **EfficientNet-B0-layout MBConv** (t=6, ~5M),
**224×224 RGB**, moderate reg (dropout 0.3, drop_path 0.1, translation), EMA + OneCycle.

**✅ RESULT: 69.32% val (single fold) — the 66% plateau is broken (+3.3 pp).** Architecture +
resolution *was* the lever, exactly as the historical 70% runs implied; our 128×64-G / plain-ResNet
detour was the regression. Crucially the run is **healthy** — gap ≈ 0 the whole way (train 68 / val 69,
no over-fit) and still improving at ep40, so there is **headroom**:
- **EMA decay 0.9995 was too high** — EMA lagged the raw model (67.4 vs 69.3) and was still rising at
  ep40. Lowered to **0.999** for the next run (should match/beat raw).
- **Still improving at 40 ep** → extended to **60**.
- **3-fold + TTA** never yet banked → ~+1–3 pp.
- **No over-fit** → more capacity (width 1.2 / B1-scale) is now likely productive (it wasn't at 128×64).

So the path forward is real: Run 24 single-fold **69.3 → est. ~72 Kaggle**; with the tuned recipe +
ensemble + TTA, low-to-mid 70s is realistic, and the 0.83 target is plausibly reachable by continuing
to scale this architecture (capacity, epochs, resolution) now that it generalizes.

**Run 24b (tuned: EMA decay 0.999, 60 ep) — ✅ 70.45% val** (EMA, ep46; est ~73 Kaggle). The EMA fix
worked exactly as predicted: EMA now **leads** the raw model (70.45 vs raw ~69) from ep28 on, adding
~+1 pp over Run 24a. Healthy until ~ep44, then a **mild over-fit tail** appears (train 75 / val 70,
gap turns +) — so ~ep46 is this single-fold config's peak; `best_state` captures it. Progression so
far: **66 → 69.3 → 70.45.** Remaining levers (now that mild over-fit is the failure mode, not
under-fit): **3-fold + TTA** (for the submission), then **more capacity** paired with stronger reg
and higher resolution.

**Run 25 — Plan 2 (compound scale-up, `07_tpu_train_effnet_b2.ipynb`):** following EfficientNet's
principle of scaling capacity + resolution + regularization *together*, B2-scale (**7.83M, 23 MBConv
blocks**, width 1.1 / depth ×1.2) @ **256×256 RGB**, dropout 0.4 / drop_path 0.2, 70 ep, EMA 0.999.
Verified to build (7.83M, forward OK at 256). Target: push the 70.45 single-fold peak into the
mid-70s, then ensemble + TTA toward 0.80. **Result: EMA peak 71.53% (ep43), single fold** — see Exp 31.

**Kaggle data-pipeline fix (`07_kaggle_effnet_b2.ipynb`):** the first Kaggle run sat at 0–5 % GPU /
200 % CPU — per-sample CPU augmentation in `__getitem__` + DataLoader workers starved both T4s.
Rewrote to **cache all images as uint8 tensors in RAM** (`cache_u8` → `XTR`/`XTE`, ~22 GiB) and do
**all augmentation on the GPU** (`gpu_aug`: per-batch translation roll ±18/±28 + per-sample
brightness/contrast/noise), indexing the cache directly with no DataLoader. GPU util → 80–95 %,
epoch time minutes → ~30–60 s. This is the canonical fast path for any future Kaggle/Colab run.

### Exp 31 — Run 25 result: scale-up gives +1.1pp; the wall is now generalization. Plan to ≥75 🔄

**Run 25 single-fold curve (B2-scale 7.83M @ 256 RGB, 70 ep, EMA 0.999):**

| ep | 20 | 30 | 34 | 40 | **43** | 44 | 45 | 47 |
|----|----|----|----|----|--------|----|----|----|
| EMA val | 64.5 | 70.1 | 71.1 | 71.3 | **71.53** | 71.4 | 71.2 | 70.9 |
| train | 61.5 | 65.6 | 67.2 | 70.0 | 72.4 | 73.4 | 74.7 | 76.5 |
| gap (tr−raw) | −3.0 | −4.5 | −3.8 | −1.3 | +0.9 | +2.0 | +3.5 | +5.6 |

**Read:** EMA peaks **71.53 % at ep43** (+1.1pp over Run 24b's 70.45). Healthy until ep42 (gap ≤ 0),
then the gap flips positive and widens to +5.6 by ep47 while EMA *declines* — the last ~25 epochs
are pure memorization (LR still 3.6e-4). Compound scaling helped, but only a little: **capacity is
no longer the lever — overfitting is.** Consistent with Exp 29 (the signal is subtle local spatial
structure; the model has more than enough capacity to fit it, and then to over-fit it).

**Why ~71.5 and not higher (single fold):** B2 at 256 has the capacity to drive train past 76 %, but
the current regularization (dropout 0.4 / drop_path 0.2 / wd 1e-3 / translation+jitter+noise) only
holds the gap to ~ep43. After that, nothing stops memorization, so the *best* epoch is the one just
before reg gives out — not a higher plateau.

**Plan to ≥75 (ranked by expected impact, all spectrogram-valid & cheap):**

| # | Lever | Why it should work | Est. gain | Status |
|---|-------|-------------------|-----------|--------|
| 1 | **3-fold ensemble** (`N_FOLDS=1→3`) | Infra already supports it; predict averages softmax. Three independent 71.5 % models decorrelate errors. This is the **single biggest untapped lever** and is free (already coded). | **+1.5–2.5pp** on the blend | TODO |
| 2 | **Stronger late reg** — wd 1e-3→2e-3, drop_path 0.2→0.3, + light **SpecAugment** (freq-mask ≤16 / time-mask ≤24 on GPU) | Directly attacks the ep44+ memorization tail so each fold peaks *higher and later* instead of decaying. Masking is README-✅ valid (axes preserved). | **+0.5–1.5pp** per fold | TODO |
| 3 | **Translation TTA at inference** — average softmax over ~5 small shifts (the only README-✅ TTA; flips/rotation are ❌) | Exp 28/29: class ≈ orientation/position of thin lines; small shifts are label-preserving and decorrelate. | **+0.3–0.7pp** | TODO |
| 4 | **Stop wasting the tail** — cosine still trains hard at ep47. EMA already captures the peak, so 70 ep is fine *if* reg (lever 2) keeps val rising; otherwise cut to ~55 ep. | Removes the pure-overfit segment. | neutral→+0.3 | TODO |

**Expected stack:** single fold ~72 (with lever 2) → 3-fold blend ~74 → +TTA ~74.5–75 val, i.e.
**Kaggle ~77** (val + 2.7). That reaches the ≥75 target with margin on Kaggle and lands at/just below
75 val. Reaching the colleagues' **0.83 Kaggle (~80 val)** is *not* in tuning range of B2 — that
implies a structurally different approach (different input encoding or a task-specific head), tracked
separately. **Levers 1–3 are the realistic path to the stated 75 goal.**

### Exp 32 — The reframe: the label is a COUNT of objects (ordinal), not a nominal class 🔄

**Decisive new information (user, 2026-05-30):** the 5 labels are the **count of signal objects**
(ordinal 1–5), the **test set may be used for semi-supervised pseudo-labeling**, and the goal is a
full campaign to **80+ Kaggle**. This is a *counting* task, which retroactively explains the entire
log: every **global** handcrafted feature scored at chance (Exp 29) because counting is **local +
nonlinear** (detect objects, then aggregate) — exactly what a CNN does and a global statistic cannot.

**Two immediate consequences:**
1. **Ordinal modeling should beat nominal softmax** — predicting "≥k" thresholds (CORN) or a regression
   scalar respects the ordering; on a count target where errors are mostly ±1, that lifts accuracy.
   (This is the "treat it as regression?" lever the user's colleagues hinted at.)
2. **Masking/cutout augmentation is now suspect** — a freq/time mask or erase box that covers a thin
   object changes the *true* count but not the label → **injected label noise**. This is the most
   likely explanation for the augmentation graveyard (Exp 22/27/28). Prefer **count-preserving** augs
   (translation, contrast/SNR, additive noise); re-test masking rather than assume it's valid.

**Phase 0 probe (`src/probe_count.py`) — can oriented matched-filters recover the count?** A 36-filter
Gabor bank (12 orientations × 3 scales) → per-image energy/peak/orientation features → RandomForest
(5-fold CV) on 3,000 train images:

| Representation → shallow model | 5-class acc |
|---|---|
| chance | 20.0% |
| Exp 29 raw G pixels (logreg) | 20.3% |
| Exp 29 naive component count | ~23% |
| **Gabor oriented features (RF)** | **29.6%** |
| (the CNN, for reference) | ~71% |

`spearman(total oriented energy, label)=+0.04`, `spearman(peak-count, label)=+0.05` — **global
aggregates still don't correlate with the count** (objects overlap, vary in slope, and sit in a
variable noise floor — confirmed by eye on a per-class montage). But oriented features beat raw pixels
by **+9.3pp**, so **oriented filtering exposes count signal that raw pixels hide**. Conclusions:
(a) the count must be **learned locally by the CNN** — a hand-counter (29.6%) is too weak to ensemble;
(b) the **oriented front-end direction is validated** enough to A/B (Phase 3); (c) the **ordinal head
is the priority** (Phase 1).

**Campaign plan:** Phase 0 probe ✅ → Phase 1 ordinal head A/B (softmax vs CORN vs regression,
`src/ab_ordinal_test.py`) 🔄 → Phase 2 count-preserving aug A/B → Phase 3 oriented front-end A/B →
Phase 4 pseudo-labeling → Phase 5 multi-arch ensemble. Each gated on exact-count accuracy before
spending Kaggle compute/submissions. **Result: Phase 1 done — see Exp 33.**

### Exp 33 — Ordinal/regression heads FALSIFIED on the exact metric ✅ (decisive negative)

**Phase 1 A/B** (`src/ab_ordinal_test.py`): identical from-scratch B0-MBConv (4.0M) @160 RGB, full
15.5k, single 80/20 split (seed 42), 36 ep, count-preserving aug — **only the head/loss/decode differ**:

| Head | Loss | Exact acc (Kaggle metric) | ±1 acc |
|------|------|---------------------------|--------|
| **softmax** | CE + label-smoothing 0.1 | **67.68%** | 89.45% |
| CORN ordinal | 4× conditional-binary "≥k" | 64.39% (−3.29) | 91.87% |
| regression | SmoothL1 → round | 60.94% (−6.74) | 93.23% |

**Decisive tradeoff:** softmax → CORN → regression, **exact accuracy falls monotonically (67.7 → 64.4
→ 60.9) while ±1 accuracy rises (89.5 → 91.9 → 93.2).** Ordinal/regression losses optimize *ordinal
proximity* — they pull predictions toward the right neighborhood but **soften the exact adjacent-count
boundary**, which is exactly what the accuracy metric punishes. **Gate failed → keep softmax.** (This
also answers the colleagues' "treat it as regression?" hint: regression helps MAE/±1, not accuracy.)

**The real lesson (redirects the campaign):** ~90% ±1 means the network *already* learns the count
ordering; the hard ~32% is the **exact boundary between adjacent counts** (is it 3 or 4 objects?).
That is a *detection-sharpness* problem (resolve/separate individual objects), not a loss problem.
→ Prioritize the **oriented-line stem** (Phase 3, sharper object detection), **ensembling**, and
**pseudo-labeling** over head engineering. (Open hypothesis from the same logic: the winning softmax
still used label-smoothing 0.1 — since smoothing also blurs the adjacent-count boundary, a cheap
follow-up is to A/B label-smoothing 0.1 vs 0.05 vs 0.0; MixUp, which softens harder, stays contraindicated.)

### Exp 34 — Hand-engineered object counting fails definitively → the count is CNN-only ✅ (negative)

Follow-up to Exp 32: if the label is a line-object count, a *line detector* should count it. Built a
Radon line-counter (`src/probe_hough.py`): Gabor max-response enhancement (suppress the variable noise
floor) → binarize → Radon transform → count well-separated sinogram peaks. Swept binarization quantile,
peak threshold, min-separation. **Best result: spearman(line-count, label) = −0.16** (weak and
*wrong-signed*); mean predicted count is flat across labels. Combined with Exp 32's Gabor-energy RF
(29.6%) and Exp 29's naive component count (~23%), **three independent hand-counters all fail.**

**Conclusion:** the object count is *not* recoverable by any global/handcrafted detector — it is an
irreducibly **learned, local, nonlinear** quantity (objects overlap, cross, and sit in heavy variable
noise; separating them to count exactly is the whole difficulty). **There is no feature-engineering
shortcut.** Every remaining lever toward ~80 is CNN-side: sharper detection (oriented stem, Phase 3),
**input resolution that separates adjacent objects** (re-test for the counting frame — Exp 23's
"resolution flat" was nominal-classification-era), **ensembling + TTA** (Phase 5), and **test-set
pseudo-labeling** (Phase 4, the single biggest untapped lever, now permitted).

### Exp 35 — Phase 2+3 A/B: masking augmentation FALSIFIED (erases objects); oriented stem 🔄

Combined A/B (`src/ab_campaign.py`), softmax head (Phase-1 winner), identical B0-MBConv (4.0M) @160
RGB, full 15.5k, single split seed 42, 36 ep. One shared `plain+safe` baseline serves both the
augmentation comparison (Phase 2) and the oriented-stem comparison (Phase 3).

| Arm | Stem | Aug | Exact acc | Δ vs SAFE |
|-----|------|-----|-----------|-----------|
| **SAFE** | plain | translation + contrast/SNR + noise | **67.68%** | — (reproduces Exp 33 softmax exactly) |
| MASK | plain | SAFE **+ SpecAugment freq/time masking** | 63.87% | **−3.81 pp** ❌ |
| GABOR-LRN | learnable Gabor | safe | 66.13% | **−1.55 pp** ❌ |
| GABOR-FIX | fixed Gabor | safe | 66.32% | **−1.36 pp** ❌ |

**Phase 2 — DECISIVE:** SpecAugment masking costs **−3.8 pp exact** (and −1.6 ±1). This confirms the
Exp 32 prediction: a freq/time mask that covers a thin signal line **deletes an object and corrupts
the count label**. It is the most likely culprit behind the historical augmentation graveyard
(Exp 22/27/28), which all predate knowing the task is counting. **Action: strip masking from the
Kaggle notebook `gpu_aug` (Run 26 currently includes it) — keep only count-preserving augs.** First
concrete recovered point of the path-to-80 campaign. ✅

**Phase 3 (oriented stem) — FAILED ❌ (both variants confirmed):** the Gabor line-detector front-end
scores **66.13% learnable / 66.32% fixed — −1.55 / −1.36 pp below SAFE** (gate was ≥ +1.0 pp). Neither
beats the plain random stem; the Gabor structure mildly constrains a stem that already learns better
filters on its own. Consistent with Phase 0 (oriented features only weakly helped globally). **Drop the
oriented stem — Phase 3 closed.** This closes the last *structural* lever: **ordinal head (Exp 33),
masking (Exp 35 Phase 2), and oriented stem (Exp 35 Phase 3) have all failed.** The path to 80+ is now
entirely **non-architectural** — pseudo-labeling (Exp 37), 3-fold ensemble + TTA, resolution, and
capacity. Weight shifts decisively to **pseudo-labeling** as the headline swing.

### Exp 36 — Run 27 on Colab: R0 fold-0 = 72.3% (count-aware config validated) 🔄

First fold of Run 27's round-0 ensemble (count-aware B2@256, no masking, SNR-varying noise, wd 1e-3 /
drop_path 0.2, softmax, 70 ep, single T4):

| ep | 30 | 40 | **46** | 50 | 53 |
|----|----|----|--------|----|----|
| EMA val | 69.8 | 71.5 | **72.29** | 71.5 | 71.1 |
| train | 63.6 | 67.0 | 71.0 | 74.4 | 77.4 |
| gap (tr−raw) | −6.2 | −4.6 | −1.3 | +2.9 | +6.3 |

**Read:** EMA peaks **72.29% at ep46** = **+0.76 pp over Run 25's 71.53** single-fold — confirms the
count-aware recipe (masking stripped + SNR-varying noise) is healthy and slightly ahead, with no
regression from the changes. Same mild overfit tail as Run 25 (gap flips positive ~ep48, EMA drifts
down; EMA/best-state correctly banks the ep46 peak — 70 ep is ~15 ep longer than needed). Folds 1–2 +
3-fold ensemble + translation TTA still to come (expected CV ~74 → ~77 Kaggle); pseudo-labeling
(`N_ROUNDS=1`) is the swing to ~80. **In progress on Colab.**

### Exp 37 — Pseudo-labeling VALIDATED offline (+1.5pp) — but the notebook's threshold recipe was broken ✅

Offline validation harness (`src/ab_pseudo.py`): carve a stratified **3,500-image proxy "test"** out of
train (true labels known but **hidden during training**, revealed only to score), run the real
self-training loop, and read the gain on both the clean val fold (the CV gate) **and** the proxy test's
true accuracy (the number that matters). EffNet B0-MBConv @160, softmax, count-preserving aug, EMA.

**Critical bug found — `TAU=0.95` selects ~0 images under label smoothing.** Training uses
`label_smoothing=0.1`, which calibrates the model away from overconfidence. Measured max-softmax
confidence percentiles on the proxy test: **p50 0.49 / p75 0.67 / p90 0.79 / p95 0.84 / p99 0.88** — the
*99th percentile* is only 0.88. So `conf > 0.95` selects **0** images and `> 0.90` selects **6**. The
planned Colab pseudo cell (`pseudo_label_cell.py` / notebook CELL 8) used **TAU=0.95** — it would have
injected an empty pseudo set and made the headline lever a **silent no-op** (the +2–4pp swing, dead on
arrival, with no error to flag it).

**Fix — top-K-per-class selection instead of an absolute threshold.** Take the most-confident *K* per
predicted class (class-balanced, robust to the LS-capped low-confidence regime), with a loose floor
(0.55) to avoid near-random injects:

| | clean val (CV gate) | proxy-test true acc | pseudo set |
|---|---|---|---|
| **R0** (real labels) | 63.33% | 61.86% | — |
| **R1** (+ top-250/class pseudo, w 0.5) | 64.00% (**+0.67**) | **63.34% (+1.49)** | 1,195 imgs @ **88.7% correct** |

**Read:** the top-K confident subset is **88.7% accurate vs the 62% base** — confidence *does* track
correctness here, so injecting it is net-positive. **+1.49 pp on the proxy test is a conservative floor**
(iid proxy = no distribution shift to exploit; the real Kaggle test may gain more). **Pseudo-labeling is
a real, validated lever — but ONLY with top-K selection, NOT the TAU=0.95 threshold.** This is the fix
that goes into the next Colab run. → Notebook changes: top-K `build_pseudo`, NaN guard, `N_ROUNDS=1`. ✅

### Exp 38 — Resolution IS a lever for counting (reverses Exp 23): S=160→224 = +2.09pp ✅

`src/ab_resolution.py` A/B (same EffNet B0-MBConv, softmax, count-preserving aug, same StratifiedKFold
first split; the only change is input size). **Exp 23 had declared resolution dead** (224 vs 128×64 =
+0.65pp on the *old ResNet*) — but that predated the counting reframe. Re-measured on the count-aware net:

| input | exact acc | Δ vs 160 |
|---|---|---|
| **S=160** | 67.68% | — |
| **S=224** | **69.77%** | **+2.09** |

**Read:** resolution is one of the **largest** single levers found. The objects are thin lines on a
natively **128×55** (H×W) grid — the time axis is only 55px, so upscaling separates close/overlapping
lines and reduces under-counting. **Important caveat: the live Colab run is already at S=256 (> 224), so
this gain is already baked into Run 27's 72.3%** — it explains why the real run beats the 160 proxy, but
is *not* additional headroom. The only *extra* resolution upside is going **above 256** (288/320), which
should be smaller (native height 128 → 256 is already 2×). **Curve resolved: S=224 69.77 → S=256 69.87
(+0.10) — saturated.** Resolution plateaus at ~224–256; the run's 256 is already optimal. **No bump.**

### Exp 39 — Phase 5 multi-arch blend: below gate (+0.35pp), single arch kept ❌

`src/ab_blend.py`: train a 2nd, deliberately distinct from-scratch net (`ResNetCounter` — plain 3×3 conv +
residual blocks, identity-start zero-init BN; no depthwise/SE, so its errors decorrelate from the EffNet's
MBConv/SE design) and ensemble it with the EffNet. Same split, S=160, 32 ep.

| model | exact acc |
|---|---|
| EffNet solo | 65.77% |
| ResNetCounter solo | 65.26% |
| best blend (w_eff 0.6) | 66.13% (**+0.35** vs best solo) |

**Read:** the contrast net trains competitively (65.3%) but the blend adds only **+0.35 pp — below the
+0.5 gate.** The two architectures are too correlated (both mid-60s CNN counters) for the average to
help much, and 3-fold + TTA already supply most of that decorrelation. **Not worth 2× training. Keep the
single EffNet architecture.**

### Campaign conclusion — every lever measured

| Lever | Verdict | Effect |
|---|---|---|
| Ordinal/CORN head (Exp 33) | ❌ | −3.3 |
| Masking aug (Exp 35) | ❌ | −3.8 |
| Oriented Gabor stem (Exp 35) | ❌ | −1.4 |
| Resolution > 256 (Exp 38) | ➖ | saturated (256 already optimal) |
| Multi-arch blend (Exp 39) | ❌ | +0.35 (< gate) |
| **Top-K pseudo-labeling (Exp 37)** | ✅ | **+1.5** (iid floor; transductive could be more) |
| 3-fold ensemble + translation TTA | ✅ | +~2 (prepped) |

**The only validated *new* gain is top-K pseudo-labeling (+1.5).** Evidence-based projection for the next
run (3-fold + TTA + top-K pseudo @256): **~76 val ≈ ~79 Kaggle**, range 78–80. **82 is reachable only if
pseudo-labeling over-delivers on the real (distribution-shifted) test vs the iid proxy, and/or a 2nd
pseudo round compounds** — neither validatable offline. The notebook is built for that best shot.

### Exp 40 — Run 27 Kaggle calibration; pseudo R1 rejected; prior-correction FALSIFIED ✅

Run 27 (vast.ai RTX 2080 Ti, N_ROUNDS=1) finished. R0 CV **71.34%**; pseudo R1 CV 71.75% **rejected by the
+0.5 gate** (only +0.41 — the iid +1.5 proxy did NOT transfer to the real test). Final = R0, submitted.

- **Kaggle public = 0.74836.** Predicted 71.34 + 2.7 = 74.0 → actual **+3.5 offset**. The +2.7 rule holds
  (slightly conservative). **Clean CV transfers to Kaggle ~1:1 → 82 Kaggle needs ~78.5 CV** (we're at 71.3,
  i.e. **+7.2 CV** needed — not a tuning gap).
- **Prior/distribution correction FALSIFIED** (`posthoc_prior.py`): the model over-predicts label-1 (test
  34.7%, val 28.8% vs 22.6% true), but IPF reweighting to the train prior moved OOF val **71.33 → 71.19
  (Δ −0.14)**. The class weights came out ≈1 (label-1 = 0.999) — the *soft* marginal is already calibrated;
  the argmax skew is genuine model uncertainty defaulting to count=1, not a correctable calibration bias.
- **Detect-then-count re-confirmed dead** (Exp 29/32/34 already): no point re-probing classical counting.

### Exp 41 — ASPECT RATIO is a real lever: native-aspect beats square stretch +3.55pp ✅ (first new structural win)

`src/ab_aspect.py`. The deployment pipeline resizes native **128(freq)×55(time)** → **square 256×256**,
stretching the time axis ~2.3× *non-uniformly*. For a counting task this smears adjacent objects together.
Clean isolation — **matched pixel budget (~16.4k), only the shape varies**, same EffNet/aug/EMA/OneCycle,
StratifiedKFold(5, seed42) first split:

| arm | exact acc |
|---|---|
| SQUARE 128×128 (16384px, distorted) | 63.10% |
| **NATIVE 196×84 (16464px, H/W=2.33)** | **66.65%** |
| **Δ (NATIVE − SQUARE)** | **+3.55** |

Led at *every* checkpoint (ep8 +7.3, ep16 +2.3, ep24 +1.9, ep32 +3.6) → not noise. **The square stretch was
costing ~3.5pp by destroying object separability along time.** This is the first lever to clear the gate
since pseudo, and it makes Exp 38's "resolution helps" finding suspect-as-confound: more square pixels may
have helped partly by *reducing* the relative stretch, not by adding genuine resolution.

**Action:** switch deployment input to **native aspect** (e.g. 256(H)×112(W), or matched-budget 391×168) —
which is also *cheaper* than square 256² — and rerun the 3-fold on the GPU. The real CV vs the 71.34 baseline
is the confirmation. This updates the "structural levers exhausted" conclusion — aspect was never tested and
it is **not** exhausted.

**Scale-up confirmation (matched ~50k px):** SQUARE 224×224 = 69.77% (exactly matches Exp 38 S=224 — harness
sanity check) vs **NATIVE 342×147 = 71.55%, Δ +1.77pp.** The effect shrinks with resolution (+3.55 @16k →
+1.77 @50k: more pixels partly compensate for the stretch) but stays well above the +0.7 gate — **confirmed
real at deployment scale.** Notably the NATIVE *single fold* (71.55) ≈ the full 3-fold SQUARE deployment CV
(71.34): one undistorted fold matches three distorted folds ensembled → genuine headroom. **Go: deploy
native aspect (matched-budget ~384×165) on the GPU.**

### Exp 42 — Run 28 live + native-aspect resolution sweep (in progress) 🔄

**Run 28 (`run27_vast.py`, native 384×165, 58 ep, any-gain pseudo gate)** confirms Exp 41 transfers to the
full B2 regime: **R0 CV 74.33% — folds [74.7, 74.0, 74.3], spread 0.7pp, all three held.** That is **+2.99pp
over Run 27 (71.34)** from the native-aspect fix alone — landing inside the projected 74–76 band and the
single biggest CV jump of the campaign. Kaggle projection ~77.8 (offset +3.5) → clears the 77 realistic
target. Pseudo R1 (top-350/class, floor 0.55, 1750 used, conf 0.768–0.956) running under the lowered
any-gain gate; adopts if CV > 74.33 by any margin.

**Follow-up A/B (`src/ab_native_res.py`, MPS) — DONE:** Exp 38's "256 is optimal" was measured on
*distorted square* images, so the resolution optimum was re-tested **at fixed native aspect** (H/W=2.327):

| native res | px | exact acc |
|---|---|---|
| 128×56 | 7k | 60.26% |
| 256×110 | 28k | 67.97% |
| **384×165** | 63k | **69.35%** (best, @ep24 lead) |

**Verdict: resolution still helps monotonically — 384×165 wins (+1.4pp over 256, +9pp over 128).** The
"256 might match 384 and be sharper" hypothesis is **falsified**: the interpolation-blur cost is real but
outweighed by the extra spatial detail. Deployment's 384×165 is the right resolution, *not* a budget-match
accident — **keep it.** Higher still (512×220) is the only untested direction but gains are clearly
diminishing (+9 → +1.4) at ~2.5× compute — deprioritized vs other levers.

**CAPACITY @ NATIVE ASPECT (`src/ab_capacity_native.py`) — DONE, second big lever ✅:** capacity was ruled
exhausted in Exp 21/23/26 — but all on *distorted square* inputs where the wall was overfitting. Native
aspect flipped the train/val gap **negative** (val > train, healthy), so "bigger just overfits" was suspect.
A/B at native 256×110:

| width | params | exact acc |
|---|---|---|
| BASE w1.0 | 4.0M | 67.97% |
| **WIDE w1.3 / depth1.2** | 10.8M | **71.42%** |

**Δ +3.45pp — capacity HELPS at native aspect, decisively.** The Exp 21/23/26 "capacity exhausted" verdict
was a square-distortion artifact: at native aspect the bigger model *generalizes* (gap stayed negative all
32 ep, no overfit). This is the 2nd structural lever after aspect ratio. **Action: bump CONFIG to
width=1.3/depth=1.2 (~10.8M scaled to deployment) for the next GPU run.** Caveat: the proxy spans 4M→10.8M;
deployment is already 7.83M (w1.1), so the *real* increment is 7.83M→10.8M (w1.1→1.3) — a fraction of
+3.45, but same direction. 

**Capacity-push (`src/ab_capacity_push.py`) — DONE, w1.3 is the deploy point:** width 1.6 (16.3M) =
**72.00%** vs w1.3 ref 71.42%, **Δ +0.58** — *below* the +0.7 promotion gate. (Note: w1.6 ran noisy and
looked behind mid-run — ep24 69.55 — but climbed in the back half to finish +0.58 ahead, so it is a *marginal
gain*, not a regression.) **Verdict: deploy w1.3 — the +0.58 from w1.6 isn't worth +50% compute/VRAM, and the
gate isn't cleared.** Capacity is effectively saturated; w1.6 stays on the shelf as a marginal backup lever
if Run 29 lands at 79.x and needs one more half-point.

**Multi-arch ensemble blend (`src/ab_ensemble_blend.py`) — DONE, FALSIFIED ❌:** EffNet_w1.3 71.71% +
DilatedCounter (dilated-residual, no depthwise/SE) 69.00% → 50-50 blend **71.06%, Δ −0.65** vs best single.
Pred correlation **0.896** (high). The distinct architecture was both weaker (−2.7pp) AND too correlated to
add complementary signal — blending dragged the strong model down. **Arch-diversity ensembling is dead here.**

**Next (`src/ab_ensemble_multires.py`): MULTI-RESOLUTION ensemble** — better diversity axis. Instead of a
weaker 2nd architecture, blend the SAME strong EffNet at two native scales (256×110 + 384×165). Both members
are strong (~71–74%) and errors should decorrelate by receptive-field scale, not by a capability gap. If
blend ≥ best single +0.5 → add a 256-scale member to deployment. Directly deployable (already train EffNet).

---

### Exp 43 — Run 30: compound scaling (448×w1.3) WORKS; multi-scale TTA FALSIFIED ✅❌

**Run 30** (`run30_vast.py`, 2-fold of 5-way split, 448×192, w1.3, N_ROUNDS=1, aggressive pseudo top-500/floor0.50).

**(1) Compound scaling transferred.** R0 CV **76.16** (folds 75.5/76.8) vs Run 28 (384, w1.1) R0 74.33 = **+1.83pp**, despite Run 30 averaging only 2 folds (usually reads *lower*). Capacity (w1.3) was dead at 384 (Run 29 +0.06) but pays off once resolution rises to 448 — exactly EfficientNet compound scaling (resolution & capacity must scale together). The res×capacity interaction is real; width is not a standalone lever but *is* one paired with resolution. Pseudo R1 promoted (base CV **76.57**).

**(2) Multi-scale TTA is HARMFUL — falsified the Run 30 design assumption.** Hypothesis: EffNet's global-pool head accepts any input size → re-scaling test images to {384,448,512} = free multi-res ensemble. **Wrong.** `tta_probe.py` measured each TTA component on the R1 held-out val folds (leakage-free, 0 submissions):

| config | fold0 | fold1 | mean | Δ vs base |
|---|---|---|---|---|
| base (448, no TTA) | 75.74 | 77.39 | **76.57** | — |
| +translation (448) | 76.52 | 77.48 | **77.00** | **+0.44** ✅ |
| +multiscale {384,448,512} | 72.32 | 77.06 | **74.69** | **−1.88** ❌ |
| full (scales × trans = `submission.csv`) | 73.23 | 77.16 | **75.20** | **−1.37** ❌ |

The head *accepting* any size ≠ features *surviving* the rescale — 448→384/512 is a feature-scale regime the model never trained on. **Translation TTA is the only valid TTA (+0.44); multi-scale must be dropped.** Run 30's auto-written `submission.csv` (full config) is −1.37 below plain 448 → **DO NOT upload it.** Corrected submission via `make_submission_corrected.py` (448 + translation only) → `submission_run30_corrected.csv`, backed by **77.00 CV** → est ~0.80–0.81 Kaggle (+4 native offset). **Lesson: always probe TTA on val folds before submitting; a mechanism being *possible* ≠ it *transfers* (cf. the capacity-proxy miss, Exp 42). If multi-scale is wanted, the model must be *trained* multi-scale (random-resize aug), not just tested so — untested.**

---

### Exp 44 — CORRECTED: public/private leaderboard split — the selected best is the model that GENERALISES ✅

**Full Kaggle history obtained (public + private). Two corrections to earlier claims:**

1. **The 0.78981 score is REAL** — I previously called it fabricated; that was wrong. It is `submission (13).csv` (public 0.78981 / private **0.78230**). My error was the *attribution*: I logged it as "Run 31," but Run 31 was never trained. Score real, attribution false.
2. **The competition is ranked on the PRIVATE split, and the selected best is `submission_run30_corrected.csv` (private 0.78690).** The higher-*public* submissions overfit the public LB.

Top of the leaderboard, both splits:

| Submission | Public | Private | Pub→Priv |
|---|---|---|---|
| `submission (13).csv` | **0.78981** | 0.78230 | −0.0075 (overfit) |
| `submission (11).csv` (≈Run 28) | 0.78836 | 0.77381 | −0.0146 (overfit) |
| **`submission_run30_corrected.csv` (SELECTED)** | 0.78763 | **0.78690** | **−0.0007 (stable)** |
| `submission (10).csv` (Run 27) | 0.74836 | 0.74424 | −0.0041 |

**The decisive insight:** `submission_run30_corrected.csv` has the best **private** score AND the smallest public→private drop (−0.0007) — it generalises. The two submissions with higher *public* scores fell −0.0075 and −0.0146 into private; selecting on public alone would have lost rank. This is why run30_corrected is the genuine best despite not topping the public board.

**Standing lessons:** (1) **Judge by the PRIVATE leaderboard / out-of-sample stability, not the public score** — a higher public number that drops hard in private is overfitting. (2) **Never log a Kaggle score without attributing it to a specific submission file** — the "Run 31 = 0.78981" error came from pinning a real score to an untrained run. The confirmed state: **best = run30_corrected (private 0.78690); native aspect ratio is the strongest confirmed lever; the path to 0.80 is a new representation (ordinal/CORN head or oriented-line stem), not more ensemble/resolution tuning** ([[label-is-object-count]]).

---

## Next Steps (Not Yet Tried)

### 🟡 Medium Impact

- ~~**MixUp augmentation**~~ ✅ — Implemented in Run 7 (`alpha=0.2`). Eliminated overfitting; peak val 65.61%.
- ~~**Larger backbone / custom architecture**~~ ✅ — Custom ResNet-18 CNN (11.2M params) implemented in `model.py` for Run 9 (in progress).
- ~~**5-fold stratified cross-validation**~~ ✅ — Implemented in Run 13.
- ~~**Custom CRNN Architecture**~~ ✅ — Implemented in Run 14.
- ~~**OneCycleLR** instead of CosineAnnealingLR~~ 🔄 — Implemented in Run 17 (`pct_start=0.15`).
- ~~**Viridis-inversion input** (recover true magnitude vs. lossy G channel)~~ ❌ — Falsified in Exp 21 (−8.4pp vs G; RGB also a tie). Input representation is not the bottleneck.
- ~~**Stochastic depth / DropPath** (structural regularizer)~~ 🔄 — Implemented in Run 17.
- ~~**EMA of weights**~~ 🔄 — Implemented in Run 17 (decay 0.999).
- ~~**Anti-aliased BlurPool downsampling**~~ 🔄 — Implemented in Run 17.
- ~~**TTA with spectrogram-valid views only**~~ 🔄 — Implemented in Run 17 (brightness/contrast, no flips).
- ~~**Larger input** (224×224 / 320 / 384)~~ ❌ — Falsified in Exp 23: 224×224 gave only +0.65pp over 128×64 at ~12× compute. Resolution is not a lever; keep 128×64.

### 🟢 Lower Impact / Longer Term

- **Ensemble multiple architectures** — train two or three distinct from-scratch designs independently; average softmax outputs.
- **Confusion matrix analysis** — visualise which class pairs are confused most, then target those pairs with augmentation.
- **Hyperparameter search** with Optuna — LR, weight decay, batch size, augmentation strength.
- ~~**Mixed precision (AMP)**~~ 🔄 — Implemented in Run 17's T4-GPU notebook (`autocast` + `GradScaler`).

### 🔧 Infrastructure

- [x] MPS (Apple Silicon GPU) support
- [x] Dataset-computed normalisation stats (cached)
- [x] Early stopping with patience counter
- [x] Per-epoch LR logging
- [x] Removed tqdm inner-loop verbosity (one line per 50 batches)
- [x] Colab GPU(T4)/TPU training notebook with AMP (`notebooks/02_tpu_train.ipynb`, Run 17)
- [x] Cached image-decode pipeline (viridis inversion → tensors, Run 17)
- [ ] `requirements.txt`
- [ ] Per-epoch checkpoint saving (not just best)
- [ ] `argparse` / `config.py` for hyperparameters
- [ ] `notebooks/02_error_analysis.ipynb` — confusion matrix, misclassified samples

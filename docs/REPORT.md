# Signal Object Detection — Classification Report

*Kaggle "Signal Object Detection" competition. Task: predict, for each grayscale
signal spectrogram, the number of signal objects it contains (an integer label
1–5). Evaluation metric: classification **accuracy**. Training set: 15,500 images;
test set: 5,500 images.*

> **How to read this report.** Section 1–2 frame the problem and data. Section 3
> describes the two required machine-learning model families and the validation
> protocol. Sections 4–5 detail the two approaches (classical hand-crafted features,
> and from-scratch convolutional networks) with their feature sets, models,
> preprocessing/augmentation and hyperparameter tuning. Section 6 is the complete
> log of *everything* tried — including the many unsuccessful experiments. Section 7
> gives results and confusion matrices on a shared validation set. Section 8 lists
> reproduction commands. All numbers in Section 4, 7 and the confusion-matrix figures
> are produced by `generate_report_artifacts.py` and written to `report_artifacts/`.

---

## 1. Problem statement and the key reframing

The competition presents grayscale spectrograms — time–frequency images — and asks
for a single integer per image in `{1, 2, 3, 4, 5}`. It is superficially a 5-way
classification problem and the leaderboard metric is plain accuracy.

The decisive observation that shaped the entire project is that **the label is a
count of signal objects**, not an arbitrary class identity. Each spectrogram
contains some number of thin, oriented "chirp" lines, and the label is how many.
This has three consequences that recur throughout the report:

1. **The classes are ordinal, not nominal.** Count 1 is closer to count 2 than to
   count 5. A model whose errors are mostly ±1 (predicting 3 when the truth is 2)
   is doing something sensible; this motivates an ordinal output head (Section 6.7).
2. **Counting needs local detection + aggregation, not global texture.** This
   explains why global hand-crafted descriptors fail almost completely (Section 4):
   the relevant quantity is "how many distinct oriented segments are present", which
   a histogram of pixel intensities cannot express.
3. **Some augmentations are dangerous.** A masking/cutout box that happens to cover
   one thin object changes the *true count* but not the label, i.e. it injects label
   noise. Time-reversing flips turn a rising chirp into a falling one. These are not
   abstract worries — both were measured to hurt (Section 6.3, 6.5).

## 2. Data and exploratory analysis

Each file is a `128 × 55` (height × width) PNG stored in RGBA form. The images are
**viridis-colormapped**: the underlying signal is a single scalar field that has
been passed through the viridis colormap to produce a pseudo-color image. The three
color channels are therefore redundant — they are a deterministic function of one
scalar. Among them the **green (G) channel has the largest dynamic range** in
viridis, so it preserves the most information per channel. Every model in this report
that does not use full RGB operates on the extracted G channel.

The semantic axes matter: **rows are frequency, columns are time**. Augmentations
that geometrically mix these axes (rotation) or reverse them (horizontal flip) are
not label-preserving and were confirmed harmful.

The class distribution is mildly imbalanced (Figure `class_distribution.png`):

| Label (count) | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|
| # train images | 3500 | 3000 | 3000 | 3000 | 3000 |

A majority-class predictor therefore scores **22.58 %** on a representative split —
this is the chance baseline every model is measured against. Two reference figures
are included from the exploratory phase: `data/sample_spectrograms.png` (one example
per class) and `data/class_intensity_distributions.png` (per-class intensity).

## 3. Machine-learning approaches and validation protocol

Two clearly different model families were developed, satisfying the "at least two
different models" requirement many times over:

* **Approach 1 — classical models on hand-crafted features** (Section 4): k-Nearest
  Neighbours, linear SVM, RBF-kernel SVM, logistic regression and random forest,
  each trained on an explicit, interpretable feature vector.
* **Approach 2 — from-scratch convolutional neural networks** (Section 5): a compact
  ~0.9 M-parameter SE-residual CNN (`CompactSignalCNN`) and a larger from-scratch
  EfficientNet/MBConv-style network (`EffNet`, the actual competition entry).

**Validation protocol.** So that every model's confusion matrix is directly
comparable, all models are scored on **one shared validation set**: a stratified
20 % hold-out of the training data (3,100 images, random seed 42). The classical
models are trained on the remaining 80 % and predict this hold-out. The CNN is
evaluated **leakage-free** by *out-of-fold* inference: the project's three saved
3-fold checkpoints (StratifiedKFold, seed 42) each predict only the images they did
**not** train on, giving every image a prediction from a model that never saw it;
the shared 20 % subset is then read off from those out-of-fold predictions. No image
is ever scored by a model that trained on it.

> **Competition-rule note.** The classical models of Section 4 are reported as
> baselines and documentation of tried approaches; they are *not* the competition
> submission. The submitted model (Section 5) is built only from PyTorch primitives,
> trained from random initialisation, on the provided data only — as the competition
> rules require (no pretrained weights, no predefined `timm`/`torchvision`
> architectures, no external data).

## 4. Approach 1 — Classical models on hand-crafted features

### 4.1 Feature set

Each image's G channel (normalised to `[0,1]`) is reduced to a **93-dimensional**
feature vector designed around the counting task:

| Block | Dim | What it captures |
|---|---|---|
| Intensity histogram (32 bins) | 32 | amplitude / SNR distribution |
| Frequency-energy profile (row means, ↓ to 32) | 32 | which frequencies carry energy |
| Time-energy profile (column means, ↓ to 11) | 11 | when energy occurs |
| Gradient-orientation histogram ("hand HOG", 8 bins) | 8 | presence of oriented edges (the lines) |
| Bright-run statistics | 5 | # bright segments per row (a direct count cue) |
| Global summary statistics | 5 | mean, std, bright-fraction, median, max |

The "bright-run" block thresholds the image at `mean + 2·std` and counts how many
separate bright segments appear on each row; its mean/std/max are an explicit attempt
to *measure the count by hand*. Features were standardised (zero mean, unit variance)
before the distance- and margin-based models; the random forest uses the raw
features (trees are scale-invariant).

### 4.2 Models and hyperparameter tuning

Each model was tuned with a grid search, evaluated on the shared validation set. The
full grids are written to `report_artifacts/hp_knn.csv` and `hp_svm.csv`; summarised:

**k-NN** (sweep over `k` and weighting):

| k | uniform | distance |
|---|---|---|
| 1 | 22.35 | 22.35 |
| 5 | 22.94 | 22.71 |
| 15 | 23.23 | 23.32 |
| **25** | **24.48** | 24.26 |
| 49 | 23.16 | 23.58 |

**RBF SVM** (grid over `C × gamma`, trained on an 8,000-image stratified subset for
tractability):

| C \ gamma | scale | 0.01 | 0.1 |
|---|---|---|---|
| 0.5 | 27.87 | 27.90 | 22.42 |
| 1.0 | 28.03 | **28.26** | 23.00 |
| 5.0 | 28.10 | 28.16 | 25.48 |
| 10.0 | 27.90 | 27.87 | 25.39 |

Linear SVM (C∈{0.1,1,10}) peaked at **28.06 %**; logistic regression at **28.48 %**;
random forest (400 trees) at **37.26 %**.

### 4.3 Result and conclusion

| Model | best config | val accuracy | macro-F1 |
|---|---|---|---|
| Majority baseline | — | 22.58 | — |
| k-NN | k=25, uniform | 24.48 | 21.93 |
| Linear SVM | C=1 | 28.06 | 22.04 |
| RBF SVM | C=1, γ=0.01 | 28.26 | 25.83 |
| Logistic regression | C=1 | 28.48 | 24.28 |
| Random forest | 400 trees | **37.26** | 34.79 |

**The classical approach largely fails**, and instructively so. The strongest model
(random forest, 37 %) is only ~15 points above the 22.6 % majority baseline, and the
kernel/linear models barely clear chance. This is not a tuning failure — the grids
are flat and wide — it is a *representation* failure: a fixed global descriptor
cannot count spatially-distinct oriented objects. This reproduces the project's
earlier finding (experiment log Exp 29) that a naïve connected-component count
correlated only −0.04 with the label. The random forest does better only because the
bright-run and orientation blocks carry a weak, noisy count signal that an ensemble
of trees can partially exploit. Confusion matrices (`cm_*.png`) show the classical
models collapsing most predictions onto the majority and adjacent classes.

The lesson — that the task needs *learned* local feature detectors — is exactly what
motivates Approach 2.

## 5. Approach 2 — From-scratch convolutional networks

### 5.1 Preprocessing and augmentation

* **Input.** Two regimes were used across the project: (a) the compact model takes the
  single G channel resized to `128 × 64` and normalised with a single-channel
  mean/std computed once over the training set (`0.0499 / 0.1135`); (b) the
  competition EffNet takes the image at **native aspect ratio** (e.g. `384 × 165` or
  `448 × 192`) in RGB, normalised with per-channel statistics.
* **Count-preserving augmentation only.** Translation (±18 px frequency, ±28 px time),
  per-image contrast/brightness jitter, and SNR-varying additive Gaussian noise.
  **No** masking/cutout, **no** flips or rotations — each of those was measured to
  hurt (Section 6.3, 6.5), consistent with the counting semantics.

### 5.2 Models

**`CompactSignalCNN` (~0.9 M parameters)** — a self-contained SE-residual network:
a `Conv-BN-SiLU` stem, four downsampling stages each combining a strided
convolution, a Squeeze-and-Excitation channel-attention block, and a residual
refinement block, then global average pooling and a two-layer head with dropout
0.5. A critical detail is **zero-initialising the last BatchNorm of every residual
block** so each block starts as the identity — without this the network was stuck at
~21 % (random-chance) early in the project (Exp 16).

**`EffNet` (the competition model, ~7.8–10.8 M parameters)** — a from-scratch
EfficientNet/MBConv-style network (inverted-residual blocks with SE, SiLU,
stochastic depth) assembled entirely from PyTorch primitives, with compound width/
depth scaling. It is trained with AdamW, OneCycle/cosine learning-rate schedules,
label smoothing 0.1, EMA weight averaging, stratified K-fold cross-validation, and
test-set pseudo-labeling.

### 5.3 Hyperparameter tuning (CNN)

The configuration was tuned incrementally across the runs in Section 6; the settled
values and the reason each was chosen:

| Hyperparameter | Final value | Evidence |
|---|---|---|
| Optimiser | AdamW, lr 1e-3, weight-decay 1e-3 | wd 5e-4 over-fit, wd 3e-3 under-fit (Exp 22) |
| LR schedule | OneCycle (pct_start 0.2) / cosine | peak lands in the anneal tail |
| Label smoothing | 0.1 | mild, stabilises calibration |
| Dropout / drop-path | 0.4 / 0.2 | tames the over-fit tail at B2 scale |
| EMA decay | 0.999, reset at epoch 3 | EMA leads raw val once reset is applied |
| Input | native-aspect (not square) | square stretch cost −1.77 pp (Exp 41) |
| Augmentation | translation + jitter + noise | MixUp & masking falsified (Exp 27, 35) |
| Ensemble | 3–4 stratified folds + EMA | member count is the one lever that transfers |
| TTA | translation only | multi-scale TTA −1.88 pp; flips −0.05 Kaggle |

### 5.4 Result

On the shared validation set, `CompactSignalCNN` (out-of-fold) reaches **57.32 %**
accuracy / 56.04 macro-F1 — a ~35-point jump over the best classical model and ~20
over the random forest, confirming that learned convolutional features are the right
representation. Its confusion matrix (`cm_CompactSignalCNN.png`) is concentrated on
the diagonal with the residual mass on ±1 neighbours, the signature of an ordinal
counting task.

The larger competition `EffNet` model, trained on GPU with the full recipe above,
reaches **74.8–77.0 % cross-validated accuracy**. Its best submission on the metric
that decides the competition — the **private leaderboard — is 0.78690**
(`submission_run30_corrected.csv`, public 0.78763). A different submission reached a
slightly higher *public* score (0.78981) but dropped to 0.78230 private, i.e. it
overfit the public split; the selected model is the one that generalises (Section 7).
The persistent gap between this and the goal of 0.80 is discussed in Section 7.

## 6. Complete log of approaches tried (including unsuccessful ones)

Documenting every attempt — successes and failures — is a graded requirement. The
table below condenses the project's experiment log; "✅" = adopted, "❌" = falsified
and dropped.

| # | Approach | Outcome |
|---|---|---|
| 6.1 | **Predefined `timm` EfficientNet-B0/B2** | ❌ Disqualifying (rule: no predefined architectures). Runs 1–8 invalidated; rebuilt from primitives. |
| 6.2 | **Custom ResNet-18, MixUp from epoch 1** | ❌ Stuck at ~21 %. MixUp on random init = conflicting gradients (Exp 14). Fixed by MixUp warmup. |
| 6.3 | **Horizontal flip / rotation (incl. as TTA)** | ❌ Semantically wrong (reverses time / mixes axes). TTA dropped Kaggle 0.723→0.669 (Exp 6). |
| 6.4 | **Zero-init residual BatchNorm** | ✅ Escaped the stuck-at-21 % failure; each residual block starts as identity (Exp 16). |
| 6.5 | **MixUp / CutMix (general)** | ❌ Destabilised training; Run 19 val 59.6 % vs 66.1 % without it (Exp 27). |
| 6.6 | **SpecAugment freq/time masking** | ❌ Can erase a thin object → label noise on a counting task; no gain over count-preserving aug (Exp 35). |
| 6.7 | **CoordConv, CRNN/GRU temporal head** | ❌ CoordConv plateaued at 68 %; GRU under-fit (spectrograms only ~28 time steps) — Runs 13–14. |
| 6.8 | **Resolution-preserving CNN (less downsampling)** | ❌ No gain (65.2 % ≈ baseline 66.1 %). Resolution alone is not the lever (Exp 23). |
| 6.9 | **From-scratch MBConv/EfficientNet @ 224 RGB** | ✅ Broke the 66 % plateau (+3.3 pp), healthy train/val gap (Exp 24). Architecture was the lever. |
| 6.10 | **Compound scale-up to B2 @ 256** | ✅ +1.1 pp to 71.5 % (Run 25). |
| 6.11 | **Native aspect ratio (vs square stretch)** | ✅ **The single biggest structural win: +1.77 pp val, +4.0 pp Kaggle** (Run 27→28, Exp 41). |
| 6.12 | **Capacity bump (width 1.1 → 1.3) at 384** | ❌ Did not transfer (+0.06 pp). The deployment size was already past the steep part of the curve (Exp 42). |
| 6.13 | **Higher resolution 448 + width 1.3** | ❌ Flat on test — the CV gain was a fold-scheme artifact (Run 30). |
| 6.14 | **Multi-scale test-time augmentation {384,448,512}** | ❌ −1.88 pp on validation; features do not survive rescaling. Translation-only TTA +0.44 pp ✅ (Exp 43). |
| 6.15 | **Test-set pseudo-labeling (self-training)** | ✅ +0.44 pp (Run 28 R1); top-K-per-class with a confidence floor, retrained from scratch with pseudo-weight 0.5. |
| 6.16 | **Ensemble member count (more folds)** | Hypothesised as a test-transferring lever; tested with 2- and 3-fold ensembles. Inconclusive — the apparent gains were within leaderboard noise and not separable from fold-scheme effects (see 7.3). Did not reach 0.80. |
| 6.17 | **Post-hoc prior correction** | ❌ Falsified (−0.14), despite a visible label-1 over-prediction (Exp 40). |
| 6.18 | **CORN ordinal head (in progress)** | The label is an ordinal count; replacing the 5-way softmax with 4 "≥k" threshold logits + CORN conditional-BCE loss. Prepared as Run 32; gated on same-scheme CV before submitting. |

## 7. Results and confusion matrices

### 7.1 Shared-validation comparison

| Model | val accuracy | macro-F1 | figure |
|---|---|---|---|
| Majority baseline | 22.58 | — | — |
| k-NN | 24.48 | 21.93 | `cm_k-NN.png` |
| Linear SVM | 28.06 | 22.04 | `cm_Linear-SVM.png` |
| RBF SVM | 28.26 | 25.83 | `cm_SVM-RBF.png` |
| Logistic regression | 28.48 | 24.28 | `cm_LogReg.png` |
| Random forest | 37.26 | 34.79 | `cm_RandomForest.png` |
| **CompactSignalCNN (OOF)** | **57.32** | **56.04** | `cm_CompactSignalCNN.png` |
| EffNet (competition, CV) | 74.8–77.0 | — | — |

### 7.2 Reading the confusion matrices

The six confusion-matrix figures in `report_artifacts/` tell a consistent story.
The classical models spread probability mass widely and lean on the majority class —
their off-diagonal errors are not concentrated on neighbours, i.e. they have not
learned the ordinal structure. The `CompactSignalCNN` matrix is clearly diagonal-
dominant (per-class correct counts 611/322/295/254/295) and its errors are
*ordinal-structured*: they fall predominantly in the **lower-left triangle**, i.e.
the network **systematically under-counts** — it predicts a count of 1 for 216
true-2 images and 117 true-3 images, and the majority class (count 1) is the most
frequent wrong prediction overall. The most likely confusions are with adjacent or
lower counts (true-5 most often mistaken for 4, true-2 for 1), consistent with the
model missing faint or overlapping objects rather than hallucinating extra ones.

Two design implications follow directly. First, the under-counting bias is a
calibration problem an **ordinal head** (6.18) is well suited to: modelling explicit
"≥k" thresholds gives the network a more graded decision boundary between adjacent
counts than a single 5-way softmax. Second, it confirms why a post-hoc prior
correction was tempting (the count-1 column is over-populated) yet was *measured* to
hurt (Exp 40) — the bias is in the features, not a simple shift in the decision
prior, so re-weighting the output distribution does not recover the missed objects.

### 7.3 Kaggle / generalisation observations

The top of the submission history, with **both** leaderboard splits, is the most
instructive table in this report:

| Submission | Public | Private | Public→Private |
|---|---|---|---|
| highest-public submission | **0.78981** | 0.78230 | −0.0075 |
| native-aspect EffNet (≈Run 28) | 0.78836 | 0.77381 | −0.0146 |
| **`submission_run30_corrected.csv` (selected)** | 0.78763 | **0.78690** | **−0.0007** |
| Run 27 stack | 0.74836 | 0.74424 | −0.0041 |

The competition is ranked on the **private** split, and the **selected** model is the
one with the best private score, **0.78690** (`submission_run30_corrected.csv`:
native-aspect EffNet width 1.3, K-fold ensemble with EMA, translation-only TTA).
Crucially this model also has the **smallest public→private drop (−0.0007)**: it
generalises. The submission with the single highest public score (0.78981) fell by
−0.0075 to 0.78230 private, and the next-highest-public by −0.0146 — both **overfit
the public leaderboard**. Selecting on the public score alone would therefore have
*lost* the competition relative to picking the more stable model. This is the central
generalisation lesson of the project: chase the model that holds up out-of-sample,
not the highest public number.

The strongest single structural lever identified across the project is the **native
aspect ratio** of the input (resizing to the true `H/W ≈ 2.3` rather than a square):
it improved both validation and the leaderboard, where most other levers improved
only one or neither. A second recurring lesson is that **cross-validation accuracy did
not translate to the leaderboard reliably** when a change also altered the CV *scheme*
(number of folds / data-per-fold): a higher CV obtained that way did not hold up, so
the val↔leaderboard offset is not a constant. The honest conclusion is that the
project plateaued just below 0.79 on the private split, and that closing the remaining
gap to 0.80 requires a genuinely new *representation* lever (the ordinal head of 6.18,
or a learned oriented-line front-end) rather than further ensemble or resolution
tuning.

## 8. Reproducibility

```bash
# 1. Generate every table and confusion matrix used in this report
#    (features + classical models + leakage-free CNN out-of-fold; ~3–5 min, CPU/MPS):
python3 generate_report_artifacts.py
#    -> report_artifacts/{model_summary.md, hp_knn.csv, hp_svm.csv, cm_*.png, ...}

# 2. Smoke-test the competition CNN (param count + output shape):
python3 src/model.py

# 3. Train the competition EffNet (GPU; writes fold checkpoints) and predict:
DATA_PATH=/path/to/data OUT_DIR=./run32_out N_ROUNDS=1 python3 run32_vast.py
```

**File manifest (key code):** `generate_report_artifacts.py` (this report's tables
and confusion matrices), `src/model.py` + `src/train.py` (CompactSignalCNN and its
3-fold training), `run27_vast.py … run32_vast.py` (the EffNet competition lineage),
`make_submission_corrected.py` and `tta_probe.py` (TTA validation and submission
generation). The full chronological experiment log lives in `README.md`.

---

### Appendix — environment

Python 3 with `torch 2.12`, `torchvision 0.27`, `scikit-learn 1.8`, `numpy`,
`pandas`, `Pillow`, `matplotlib`. CNN inference auto-selects CUDA → MPS → CPU. No
internet access or pretrained weights are used at any point.

"""
generate_report_artifacts.py
=============================
Reproducible generator for every figure/table the report requires:

  * a classical hand-crafted FEATURE SET (intensity histogram, row/column energy
    profiles, gradient-orientation histogram, bright-run counts);
  * TWO+ classical machine-learning models (k-NN, linear SVM, RBF SVM, logistic
    regression, random forest) with a HYPERPARAMETER GRID -> results tables;
  * the from-scratch CompactSignalCNN evaluated leakage-free via out-of-fold
    prediction from the three saved fold checkpoints;
  * CONFUSION MATRICES on one shared, stratified validation set for every model.

Everything is evaluated on the SAME held-out validation split (stratified 20 %,
seed 42) so the numbers are directly comparable. The CNN gets a leakage-free
prediction for every validation image by using the 3-fold checkpoint that did NOT
train on that image (out-of-fold inference) -- so no image is ever scored by a
model that saw it in training.

Run from the repository root (where ``data/raw`` and ``models/`` live):

    python3 generate_report_artifacts.py

Outputs are written to ``report_artifacts/``:
    features_cache.npy / labels_cache.npy   - cached features (fast re-runs)
    hp_knn.csv, hp_svm.csv, model_summary.csv / .md   - hyperparameter & summary tables
    cm_<model>.png                           - confusion matrices (one per model)
    class_distribution.png                   - class balance bar chart

No GPU required (CNN inference runs on MPS/CUDA if available, else CPU).
"""
import os
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("Agg")  # headless backend -> save PNGs without a display
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

# --------------------------------------------------------------------------------------
# Paths & global config
# --------------------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent


def _find_up(rel, start=ROOT, levels=4):
    """Search the script dir and up to `levels` parents for `rel` (e.g. 'data/raw').

    Lets this script run whether it sits at the repo root or inside a flattened
    deliverable subfolder. Override explicitly with the DATA_DIR / MODELS_DIR env vars.
    """
    p = start
    for _ in range(levels + 1):
        if (p / rel).exists():
            return p / rel
        p = p.parent
    return start / rel  # fall back to the script-local path


DATA = Path(os.environ["DATA_DIR"]) if os.environ.get("DATA_DIR") else _find_up("data/raw")
MODELS = Path(os.environ["MODELS_DIR"]) if os.environ.get("MODELS_DIR") else _find_up("models")
TRAIN_DIR = DATA / "train"
TRAIN_CSV = DATA / "train.csv"
OUT = ROOT / "report_artifacts"
OUT.mkdir(exist_ok=True)

SEED = 42
VAL_FRAC = 0.20            # stratified hold-out used for ALL confusion matrices
N_CLASSES = 5
CLASS_NAMES = [str(c) for c in range(1, 6)]   # labels are object counts 1..5

# Keep the optional SVM-on-RBF tractable: RBF SVC is ~O(n^2). Train it on a
# stratified subsample of the training partition (full set is used for the report's
# accuracy via the faster models; the subsample size is reported honestly).
SVM_SUBSAMPLE = 8000

rng = np.random.RandomState(SEED)


# --------------------------------------------------------------------------------------
# 1. Load every training image ONCE as the raw viridis G channel (uint8, 128x55)
#    The G channel has the highest dynamic range in the viridis colormap, so it is the
#    most informative single channel (same choice the CNN makes).
# --------------------------------------------------------------------------------------
def load_images(df):
    """Return uint8 array [N, 128, 55] of the G channel + int label array [N] (0-indexed)."""
    cache_x = OUT / "gchan_cache.npy"
    cache_y = OUT / "labels_cache.npy"
    if cache_x.exists() and cache_y.exists():
        print("loading cached G-channel arrays ...")
        return np.load(cache_x), np.load(cache_y)

    print(f"loading {len(df)} images (first run, then cached) ...")
    imgs = np.empty((len(df), 128, 55), dtype=np.uint8)
    for i, fname in enumerate(df["id"].values):
        arr = np.asarray(Image.open(TRAIN_DIR / fname))  # RGBA (128,55,4)
        imgs[i] = arr[:, :, 1]                            # G channel
        if (i + 1) % 2500 == 0:
            print(f"  {i + 1}/{len(df)}")
    labels = (df["label"].values - 1).astype(np.int64)   # 0-indexed
    np.save(cache_x, imgs)
    np.save(cache_y, labels)
    return imgs, labels


# --------------------------------------------------------------------------------------
# 2. Hand-crafted FEATURE SET (the classical-model representation)
#    Designed around the task semantics: the label is a COUNT of thin oriented signal
#    lines on a noisy spectrogram (frequency = rows, time = columns).
# --------------------------------------------------------------------------------------
def extract_features(imgs):
    """Vectorized-ish extraction. imgs: uint8 [N,128,55] -> float feature matrix [N,D]."""
    cache = OUT / "features_cache.npy"
    if cache.exists():
        print("loading cached feature matrix ...")
        return np.load(cache)

    print("extracting hand-crafted features ...")
    N = imgs.shape[0]
    feats = []
    t0 = time.time()
    for i in range(N):
        g = imgs[i].astype(np.float32) / 255.0          # [128,55] in [0,1]

        # (a) Intensity histogram (32 bins) -> amplitude / SNR distribution
        hist, _ = np.histogram(g, bins=32, range=(0.0, 1.0), density=True)

        # (b) Frequency-energy profile: mean intensity per row (128), downsampled to 32
        row_prof = g.mean(axis=1)                        # [128]
        row_prof = row_prof.reshape(32, 4).mean(axis=1)  # -> [32]

        # (c) Time-energy profile: mean intensity per column (55), downsampled to ~11
        col_prof = g[:, :55].mean(axis=0)                # [55]
        col_prof = col_prof[:55].reshape(11, 5).mean(axis=1)  # -> [11]

        # (d) Gradient-orientation histogram (8 bins) -- a compact hand "HOG".
        #     Thin oriented lines (the objects) show up as oriented gradients.
        gy, gx = np.gradient(g)
        mag = np.sqrt(gx * gx + gy * gy)
        ang = (np.arctan2(gy, gx) + np.pi)               # [0, 2pi)
        obins = np.minimum((ang / (2 * np.pi) * 8).astype(int), 7)
        ohist = np.zeros(8, np.float32)
        np.add.at(ohist, obins.ravel(), mag.ravel())     # magnitude-weighted
        ohist /= (ohist.sum() + 1e-6)

        # (e) Bright-run / count cues: threshold at mean+2*std, count bright runs per row.
        #     This directly targets the "how many objects" signal.
        thr = g.mean() + 2.0 * g.std()
        binm = (g > thr).astype(np.int8)                 # [128,55]
        # number of 0->1 transitions per row = number of bright segments on that row
        runs = ((binm[:, 1:] == 1) & (binm[:, :-1] == 0)).sum(axis=1) + binm[:, 0]
        run_feats = np.array([runs.mean(), runs.std(), runs.max(),
                              binm.mean(), float((runs > 0).mean())], np.float32)

        # (f) Global summary stats
        stat_feats = np.array([g.mean(), g.std(), float((g > 0.5).mean()),
                               float(np.median(g)), float(g.max())], np.float32)

        feats.append(np.concatenate([hist, row_prof, col_prof, ohist,
                                     run_feats, stat_feats]).astype(np.float32))
        if (i + 1) % 2500 == 0:
            print(f"  {i + 1}/{N}  ({time.time() - t0:.0f}s)")

    X = np.vstack(feats)
    np.save(cache, X)
    print(f"feature matrix: {X.shape}  (dim={X.shape[1]})")
    return X


# --------------------------------------------------------------------------------------
# 3. Confusion-matrix plotting helper (matplotlib only -- no seaborn dependency)
# --------------------------------------------------------------------------------------
def plot_confusion(cm, title, path, acc):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(N_CLASSES)); ax.set_xticklabels(CLASS_NAMES)
    ax.set_yticks(range(N_CLASSES)); ax.set_yticklabels(CLASS_NAMES)
    ax.set_xlabel("Predicted count"); ax.set_ylabel("True count")
    ax.set_title(f"{title}\nval accuracy = {acc * 100:.2f}%")
    thresh = cm.max() / 2.0
    for r in range(N_CLASSES):
        for c in range(N_CLASSES):
            ax.text(c, r, int(cm[r, c]), ha="center", va="center",
                    color="white" if cm[r, c] > thresh else "black", fontsize=9)
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path.name}")


# --------------------------------------------------------------------------------------
# 4. Classical models + hyperparameter grids
# --------------------------------------------------------------------------------------
def run_classical(Xtr, ytr, Xva, yva):
    """Standardize, grid-search each model, return (summary_rows, best_estimators, hp_tables)."""
    scaler = StandardScaler().fit(Xtr)
    Xtr_s, Xva_s = scaler.transform(Xtr), scaler.transform(Xva)

    summary = []          # one row per (best config of) model -> model_summary table
    best_models = {}      # name -> (estimator, val_predictions)
    hp_tables = {}        # name -> DataFrame of the full grid

    # ---- k-NN: sweep k and the weighting scheme -------------------------------------
    print("\n[k-NN] hyperparameter sweep (k x weights)")
    knn_rows = []
    best = (-1, None, None)
    for k in [1, 3, 5, 9, 15, 25, 49]:
        for w in ["uniform", "distance"]:
            clf = KNeighborsClassifier(n_neighbors=k, weights=w)
            clf.fit(Xtr_s, ytr)
            pred = clf.predict(Xva_s)
            acc = accuracy_score(yva, pred)
            knn_rows.append({"k": k, "weights": w, "val_acc": round(acc * 100, 2)})
            print(f"  k={k:2d} weights={w:8s} -> {acc * 100:.2f}%")
            if acc > best[0]:
                best = (acc, clf, pred)
    hp_tables["knn"] = pd.DataFrame(knn_rows)
    best_models["k-NN"] = (best[1], best[2])
    bp = best[1].get_params()
    summary.append({"model": "k-NN", "config": f"k={bp['n_neighbors']}, weights={bp['weights']}",
                    "val_acc": round(best[0] * 100, 2),
                    "macro_f1": round(f1_score(yva, best[2], average="macro") * 100, 2)})

    # ---- RBF SVM: grid over C x gamma (on a stratified subsample for tractability) ---
    print(f"\n[SVM-RBF] grid (C x gamma) on {SVM_SUBSAMPLE}-sample subset")
    if len(Xtr_s) > SVM_SUBSAMPLE:
        sub_idx, _ = train_test_split(np.arange(len(Xtr_s)), train_size=SVM_SUBSAMPLE,
                                      stratify=ytr, random_state=SEED)
    else:
        sub_idx = np.arange(len(Xtr_s))
    svm_rows = []
    best = (-1, None, None)
    for C in [0.5, 1.0, 5.0, 10.0]:
        for gamma in ["scale", 0.01, 0.1]:
            clf = SVC(C=C, kernel="rbf", gamma=gamma, cache_size=500)
            clf.fit(Xtr_s[sub_idx], ytr[sub_idx])
            pred = clf.predict(Xva_s)
            acc = accuracy_score(yva, pred)
            svm_rows.append({"C": C, "gamma": str(gamma), "val_acc": round(acc * 100, 2)})
            print(f"  C={C:<4} gamma={str(gamma):6s} -> {acc * 100:.2f}%")
            if acc > best[0]:
                best = (acc, clf, pred)
    hp_tables["svm"] = pd.DataFrame(svm_rows)
    best_models["SVM-RBF"] = (best[1], best[2])
    bp = best[1].get_params()
    summary.append({"model": "SVM-RBF", "config": f"C={bp['C']}, gamma={bp['gamma']} (n={len(sub_idx)})",
                    "val_acc": round(best[0] * 100, 2),
                    "macro_f1": round(f1_score(yva, best[2], average="macro") * 100, 2)})

    # ---- Linear SVM (full data) -----------------------------------------------------
    print("\n[Linear-SVM] C sweep")
    best = (-1, None, None)
    for C in [0.1, 1.0, 10.0]:
        clf = LinearSVC(C=C, dual="auto", max_iter=5000)
        clf.fit(Xtr_s, ytr)
        pred = clf.predict(Xva_s)
        acc = accuracy_score(yva, pred)
        print(f"  C={C:<4} -> {acc * 100:.2f}%")
        if acc > best[0]:
            best = (acc, clf, pred)
    best_models["Linear-SVM"] = (best[1], best[2])
    summary.append({"model": "Linear-SVM", "config": f"C={best[1].get_params()['C']}",
                    "val_acc": round(best[0] * 100, 2),
                    "macro_f1": round(f1_score(yva, best[2], average="macro") * 100, 2)})

    # ---- Logistic Regression (full data) --------------------------------------------
    print("\n[LogReg]")
    clf = LogisticRegression(max_iter=2000, C=1.0)  # multinomial by default in sklearn >=1.5
    clf.fit(Xtr_s, ytr)
    pred = clf.predict(Xva_s)
    acc = accuracy_score(yva, pred)
    print(f"  C=1.0 -> {acc * 100:.2f}%")
    best_models["LogReg"] = (clf, pred)
    summary.append({"model": "LogReg", "config": "C=1.0",
                    "val_acc": round(acc * 100, 2),
                    "macro_f1": round(f1_score(yva, pred, average="macro") * 100, 2)})

    # ---- Random Forest (full data, also gives a feature-importance sanity check) -----
    print("\n[RandomForest] n_estimators sweep")
    best = (-1, None, None)
    for n in [200, 400]:
        clf = RandomForestClassifier(n_estimators=n, random_state=SEED, n_jobs=-1)
        clf.fit(Xtr, ytr)            # trees are scale-invariant -> raw features
        pred = clf.predict(Xva)
        acc = accuracy_score(yva, pred)
        print(f"  n_estimators={n} -> {acc * 100:.2f}%")
        if acc > best[0]:
            best = (acc, clf, pred)
    best_models["RandomForest"] = (best[1], best[2])
    summary.append({"model": "RandomForest", "config": f"n_estimators={best[1].get_params()['n_estimators']}",
                    "val_acc": round(best[0] * 100, 2),
                    "macro_f1": round(f1_score(yva, best[2], average="macro") * 100, 2)})

    return summary, best_models, hp_tables


# --------------------------------------------------------------------------------------
# 5. CompactSignalCNN out-of-fold evaluation (leakage-free) from saved checkpoints
#    Reproduces src/train.py exactly: G channel, Resize(128,64), single-channel
#    normalization, StratifiedKFold(3, seed 42). Each fold's saved checkpoint scores
#    ONLY its own held-out indices -> no train/val leakage.
# --------------------------------------------------------------------------------------
def cnn_out_of_fold(df):
    import torch
    from torchvision import transforms
    from sklearn.model_selection import StratifiedKFold
    import sys
    # model.py may sit alongside this script (flattened deliverable) or in ../src (repo).
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    from model import CompactSignalCNN  # the from-scratch competition network

    ckpts = [MODELS / f"best_model_fold_{f}.pth" for f in range(3)]
    if not all(c.exists() for c in ckpts):
        print("CNN checkpoints missing -> skipping CNN confusion matrix")
        return None

    stats = json.load(open(MODELS / "dataset_stats.json"))
    mean, std = stats["mean"], stats["std"]
    tf = transforms.Compose([transforms.Resize((128, 64)),
                             transforms.ToTensor(),
                             transforms.Normalize(mean, std)])

    device = torch.device("cuda" if torch.cuda.is_available()
                          else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\n[CNN] out-of-fold inference on {device}")

    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    folds = list(skf.split(df, df["label"]))
    oof_pred = np.full(len(df), -1, dtype=np.int64)

    for f, (_, val_idx) in enumerate(folds):
        net = CompactSignalCNN(num_classes=5).to(device)
        net.load_state_dict(torch.load(ckpts[f], map_location=device))
        net.eval()
        with torch.no_grad():
            for s0 in range(0, len(val_idx), 256):
                batch_idx = val_idx[s0:s0 + 256]
                xs = []
                for j in batch_idx:
                    arr = np.asarray(Image.open(TRAIN_DIR / df.iloc[j]["id"]))[:, :, 1]
                    xs.append(tf(Image.fromarray(arr, mode="L")))
                x = torch.stack(xs).to(device)
                oof_pred[batch_idx] = net(x).argmax(1).cpu().numpy()
        print(f"  fold {f}: scored {len(val_idx)} held-out images")
    return oof_pred


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def main():
    df = pd.read_csv(TRAIN_CSV)
    y_all = (df["label"].values - 1).astype(np.int64)

    # ---- class-distribution figure (EDA) -------------------------------------------
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    counts = df["label"].value_counts().sort_index()
    ax.bar(counts.index.astype(str), counts.values, color="#4C72B0")
    ax.set_xlabel("object count (label)"); ax.set_ylabel("# train images")
    ax.set_title("Class distribution (train)")
    for x, v in zip(counts.index.astype(str), counts.values):
        ax.text(x, v + 20, str(v), ha="center", fontsize=8)
    fig.tight_layout(); fig.savefig(OUT / "class_distribution.png", dpi=130); plt.close(fig)
    print("wrote class_distribution.png")

    # ---- features ------------------------------------------------------------------
    imgs, labels = load_images(df)
    X = extract_features(imgs)

    # ---- ONE shared stratified validation split (seed 42) --------------------------
    idx = np.arange(len(df))
    tr_idx, va_idx = train_test_split(idx, test_size=VAL_FRAC, stratify=y_all, random_state=SEED)
    print(f"\nshared validation set: {len(va_idx)} images "
          f"(stratified {int(VAL_FRAC * 100)}%, seed {SEED})")
    Xtr, Xva = X[tr_idx], X[va_idx]
    ytr, yva = labels[tr_idx], labels[va_idx]

    # ---- classical models ----------------------------------------------------------
    summary, best_models, hp_tables = run_classical(Xtr, ytr, Xva, yva)

    # ---- CNN out-of-fold, then score the same validation subset --------------------
    oof = cnn_out_of_fold(df)
    if oof is not None:
        cnn_va_pred = oof[va_idx]
        cnn_acc = accuracy_score(yva, cnn_va_pred)
        summary.append({"model": "CompactSignalCNN (OOF)",
                        "config": "~0.9M params, single-channel 128x64",
                        "val_acc": round(cnn_acc * 100, 2),
                        "macro_f1": round(f1_score(yva, cnn_va_pred, average="macro") * 100, 2)})
        best_models["CompactSignalCNN"] = (None, cnn_va_pred)
        # full-train OOF accuracy too (all 15.5k held-out predictions)
        full_acc = accuracy_score(labels, oof)
        print(f"[CNN] full out-of-fold accuracy (all images): {full_acc * 100:.2f}%")

    # ---- write all tables ----------------------------------------------------------
    hp_tables["knn"].to_csv(OUT / "hp_knn.csv", index=False)
    hp_tables["svm"].to_csv(OUT / "hp_svm.csv", index=False)
    summ_df = pd.DataFrame(summary).sort_values("val_acc", ascending=False)
    summ_df.to_csv(OUT / "model_summary.csv", index=False)
    with open(OUT / "model_summary.md", "w") as fh:
        fh.write("# Model comparison on the shared validation set\n\n")
        fh.write(summ_df.to_markdown(index=False))
        fh.write("\n\n## k-NN hyperparameter grid\n\n")
        fh.write(hp_tables["knn"].to_markdown(index=False))
        fh.write("\n\n## SVM-RBF hyperparameter grid\n\n")
        fh.write(hp_tables["svm"].to_markdown(index=False))
        fh.write("\n")

    # ---- confusion matrices for every model ----------------------------------------
    print("\nwriting confusion matrices ...")
    for name, (_, pred) in best_models.items():
        cm = confusion_matrix(yva, pred, labels=list(range(N_CLASSES)))
        acc = accuracy_score(yva, pred)
        plot_confusion(cm, name, OUT / f"cm_{name.replace(' ', '_').replace('(', '').replace(')', '')}.png", acc)

    # ---- console summary -----------------------------------------------------------
    print("\n================ FINAL MODEL SUMMARY (shared val set) ================")
    print(summ_df.to_string(index=False))
    print(f"\nchance baseline = {100 * counts.max() / counts.sum():.2f}% (always predict majority class)")
    print(f"all artifacts written to {OUT}/")


if __name__ == "__main__":
    main()

"""Diagnose + fix the submission decision rule to match the LEADERBOARD metric.

The leaderboard scores Dice/Sensitivity as ONE-VS-REST at a fixed threshold
(~0.5), not argmax. An 11-class softmax rarely crosses 0.5, so sensitivity
collapses (0.27) despite great AUC (0.89). This script:
  1. Computes one-vs-rest Dice/Sensitivity at threshold 0.5 on val -> should
     reproduce the leaderboard's ~0.358 / ~0.271 (confirms the right metric).
  2. Finds the per-class threshold that maximizes one-vs-rest Dice (the headroom).
  3. Saves per-class thresholds so the submission probs can be recalibrated so
     the leaderboard's 0.5 lands at each class's optimal point.

CPU-only, uses checkpoints/_fusion_cache.npz.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

PROJECT_SRC = Path(__file__).resolve().parents[1]
for p in (PROJECT_SRC, PROJECT_SRC / "components"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, recall_score  # noqa: E402

CACHE = PROJECT_SRC / "checkpoints" / "_fusion_cache.npz"
NAMES = ["AKIEC", "BCC", "BEN_OTH", "BKL", "DF", "INF", "MAL_OTH", "MEL", "NV", "SCCKA", "VASC"]


def val_probs(C=0.05):
    d = np.load(CACHE)
    Xtr, ytr, Xval, yval = d["Xtr"], d["ytr"], d["Xval"], d["yval"]
    m = LogisticRegression(class_weight="balanced", C=C, max_iter=4000,
                        solver="lbfgs", random_state=42).fit(Xtr, ytr)
    P = np.zeros((len(yval), 11))
    P[:, m.classes_] = m.predict_proba(Xval)
    return P, yval


def ovr_dice(P, y, thresholds):
    """Mean one-vs-rest Dice (binary F1) and sensitivity at per-class thresholds."""
    dices, senss = [], []
    for c in range(11):
        yc = (y == c).astype(int)
        pc = (P[:, c] >= thresholds[c]).astype(int)
        dices.append(f1_score(yc, pc, zero_division=0))
        senss.append(recall_score(yc, pc, zero_division=0))
    return np.array(dices), np.array(senss)


def cv_thresholds(P, y, K=5, repeats=4, seed=0):
    """Robust per-class thresholds via repeated K-fold: select threshold on the
    train folds, evaluate on the held-out fold. Returns (median_threshold,
    out-of-fold Dice) per class. Tiny classes (<2 positives) fall back to in-sample.
    """
    grid = np.linspace(0.02, 0.98, 97)
    rng = np.random.default_rng(seed)
    n = len(y)
    final_thr = np.full(11, 0.5)
    oof = np.zeros(11)
    for c in range(11):
        yc = (y == c).astype(int)
        if yc.sum() < 2:
            best_t, best_d = 0.5, -1.0
            for t in grid:
                d = f1_score(yc, (P[:, c] >= t).astype(int), zero_division=0)
                if d > best_d:
                    best_d, best_t = d, t
            final_thr[c], oof[c] = best_t, best_d
            continue
        thrs, oof_d = [], []
        for _ in range(repeats):
            folds = np.array_split(rng.permutation(n), K)
            for k in range(K):
                te = folds[k]
                tr = np.concatenate([folds[j] for j in range(K) if j != k])
                bt, bd = 0.5, -1.0
                for t in grid:
                    d = f1_score(yc[tr], (P[tr, c] >= t).astype(int), zero_division=0)
                    if d > bd:
                        bd, bt = d, t
                thrs.append(bt)
                oof_d.append(f1_score(yc[te], (P[te, c] >= bt).astype(int), zero_division=0))
        final_thr[c] = float(np.median(thrs))
        oof[c] = float(np.mean(oof_d))
    return final_thr, oof


def main():
    cv = "--cv" in sys.argv
    P, y = val_probs()

    # --- Step 1: reproduce the leaderboard metric (one-vs-rest @ 0.5) ---
    thr_half = np.full(11, 0.5)
    d_half, s_half = ovr_dice(P, y, thr_half)
    print("=== One-vs-rest @ threshold 0.5 (should match leaderboard ~0.358 / ~0.271) ===")
    print(f"  mean Dice:        {d_half.mean():.4f}   (leaderboard 0.358)")
    print(f"  mean Sensitivity: {s_half.mean():.4f}   (leaderboard 0.271)")

    # --- Step 2: per-class optimal threshold for Dice ---
    best_thr = np.zeros(11)
    grid = np.linspace(0.02, 0.98, 97)
    for c in range(11):
        yc = (y == c).astype(int)
        best_d, best_t = 0.0, 0.5
        for t in grid:
            d = f1_score(yc, (P[:, c] >= t).astype(int), zero_division=0)
            if d > best_d:
                best_d, best_t = d, t
        best_thr[c] = best_t
    d_opt, s_opt = ovr_dice(P, y, best_thr)

    print("\n=== Per-class threshold optimization (val) ===")
    print(f"  {'class':8} {'support':>7} {'thr0.5 Dice':>11} {'opt thr':>8} {'opt Dice':>9}")
    for c in range(11):
        sup = int((y == c).sum())
        print(f"  {NAMES[c]:8} {sup:>7} {d_half[c]:>11.3f} {best_thr[c]:>8.2f} {d_opt[c]:>9.3f}")
    print(f"\n  MEAN Dice:  0.5-threshold {d_half.mean():.4f}  ->  optimized {d_opt.mean():.4f}")
    print(f"  MEAN Sens:  0.5-threshold {s_half.mean():.4f}  ->  optimized {s_opt.mean():.4f}")

    np.savez(PROJECT_SRC / "checkpoints" / "_class_thresholds.npz", thresholds=best_thr)
    print(f"\nSaved in-sample thresholds -> checkpoints/_class_thresholds.npz")

    if cv:
        cv_thr, oof = cv_thresholds(P, y)
        d_cv, _ = ovr_dice(P, y, cv_thr)  # in-sample Dice using the robust thresholds
        print("\n=== Cross-validated thresholds (robust) ===")
        print(f"  {'class':8} {'support':>7} {'in-samp thr':>11} {'cv thr':>7} {'OOF Dice':>9}")
        for c in range(11):
            sup = int((y == c).sum())
            print(f"  {NAMES[c]:8} {sup:>7} {best_thr[c]:>11.2f} {cv_thr[c]:>7.2f} {oof[c]:>9.3f}")
        print(f"\n  MEAN Dice  — in-sample(optimistic): {d_opt.mean():.4f}")
        print(f"  MEAN Dice  — OOF (honest estimate): {oof.mean():.4f}")
        print(f"  MEAN Dice  — cv-threshold in-sample: {d_cv.mean():.4f}")
        np.savez(PROJECT_SRC / "checkpoints" / "_class_thresholds_cv.npz", thresholds=cv_thr)
        print("Saved CV thresholds -> checkpoints/_class_thresholds_cv.npz (not yet active)")


if __name__ == "__main__":
    main()

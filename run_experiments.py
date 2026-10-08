"""Run every experiment and save the results to results/.

Usage (from the repository root):
    python scripts/download_data.py     # once, fetches the Bonn dataset
    python run_experiments.py           # ~5-10 minutes on a laptop
    python make_figures.py              # draws the charts into figures/
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import StratifiedGroupKFold

from eeg.data import SET_ORDER, bandpass, load_bonn, make_windows
from eeg.experiments import (TASKS, cross_validate, detector_transfer, make_models,
                             task_data)
from eeg.features import AMPLITUDE_FEATURES, FEATURE_NAMES, SHAPE_FEATURES, extract_features

warnings.filterwarnings("ignore", category=UserWarning)

SEEDS = (0, 1, 2)       # 3 repeats of 5-fold cross-validation
N_SPLITS = 5


def summarize(task, model_name, split, res, extra=None):
    row = {"task": task, "model": model_name, "split": split}
    for k, v in res.items():
        if k == "confusion_matrix":
            continue
        row[f"{k}_mean"] = float(np.mean(v))
        row[f"{k}_std"] = float(np.std(v))
    row.update(extra or {})
    return row


def main(data_dir, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # 1. Load, filter, window, featurize
    signals, set_labels, rec_ids = load_bonn(data_dir)
    filtered = bandpass(signals)
    windows, window_sets, groups = make_windows(filtered, set_labels, rec_ids, window_sec=2.0)
    X = extract_features(windows)
    print(f"{len(signals)} recordings -> {len(windows)} windows x {X.shape[1]} features")
    pd.DataFrame(X, columns=FEATURE_NAMES).assign(set=window_sets, recording=groups) \
        .to_csv(out / "features.csv", index=False)

    # 2. Every model on every task, honest (grouped) and leaky splits
    rows, confusion = [], {}
    for task in TASKS:
        Xt, y, g, _ = task_data(task, X, window_sets, groups)
        for name, model in make_models().items():
            for split in ("grouped", "leaky"):
                res = cross_validate(model, Xt, y, g, split=split, n_splits=N_SPLITS, seeds=SEEDS)
                rows.append(summarize(task, name, split, res))
                if task == "five_class" and split == "grouped":
                    confusion[name] = res["confusion_matrix"].tolist()
            r = rows[-2]
            print(f"  {task:22s} {name:22s} honest BA={r['balanced_accuracy_mean']:.3f} "
                  f"leaky BA={rows[-1]['balanced_accuracy_mean']:.3f}")
    pd.DataFrame(rows).to_csv(out / "cv_results.csv", index=False)
    (out / "confusion_five_class.json").write_text(
        json.dumps({"labels": SET_ORDER, "matrices": confusion}, indent=1))

    # 3. Which features matter? Same model with all / shape-only / amplitude-only features
    model = make_models()["Gradient boosting"]
    subsets = {"All features": FEATURE_NAMES, "Shape only": SHAPE_FEATURES,
               "Amplitude only": AMPLITUDE_FEATURES, "Amplitude (std) alone": ["log_std"]}
    rows = []
    for task in TASKS:
        Xt, y, g, _ = task_data(task, X, window_sets, groups)
        for label, cols in subsets.items():
            idx = [FEATURE_NAMES.index(c) for c in cols]
            res = cross_validate(model, Xt[:, idx], y, g, n_splits=N_SPLITS, seeds=SEEDS)
            rows.append(summarize(task, "Gradient boosting", "grouped", res, {"features": label}))
    pd.DataFrame(rows).to_csv(out / "feature_sets.csv", index=False)
    print("  feature-set comparison done")

    # 4. Shortcut test: train a detector on the textbook split, test it on everyone
    rows = []
    detectors = {"Trained on healthy (A) vs seizure": ["Z"],
                 "Trained on all healthy (A+B) vs seizure": ["Z", "O"],
                 "Trained on all non-seizure (A-D) vs seizure": ["Z", "O", "N", "F"]}
    for det_name, neg in detectors.items():
        for model_name, m in make_models().items():
            rates = detector_transfer(m, X, window_sets, groups, neg, n_splits=N_SPLITS, seeds=SEEDS)
            for s, v in rates.items():
                rows.append({"detector": det_name, "model": model_name, "test_set": s,
                             "predicted_seizure_rate_mean": float(np.mean(v)),
                             "predicted_seizure_rate_std": float(np.std(v))})
    pd.DataFrame(rows).to_csv(out / "detector_transfer.csv", index=False)
    print("  detector transfer done")

    # 5. Permutation importance on held-out recordings for two key tasks
    rows = []
    for task in ("seizure_vs_interictal", "zone_vs_opposite"):
        Xt, y, g, _ = task_data(task, X, window_sets, groups)
        for seed in SEEDS:
            cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=seed)
            for train, test in cv.split(Xt, y, g):
                m = clone(model).set_params(random_state=seed).fit(Xt[train], y[train])
                pi = permutation_importance(m, Xt[test], y[test], scoring="balanced_accuracy",
                                            n_repeats=10, random_state=seed)
                for f, imp in zip(FEATURE_NAMES, pi.importances_mean):
                    rows.append({"task": task, "feature": f, "importance": imp})
    imp = pd.DataFrame(rows).groupby(["task", "feature"])["importance"].agg(["mean", "std"]).reset_index()
    imp.to_csv(out / "feature_importance.csv", index=False)
    print("  permutation importance done")

    print(f"All done in {time.time() - t0:.0f} s. Results are in {out}/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data/bonn")
    ap.add_argument("--out-dir", default="results")
    args = ap.parse_args()
    main(args.data_dir, args.out_dir)

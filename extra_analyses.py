"""Extra beginner-friendly analyses used in the write-up.

    python extra_analyses.py        # about 2 minutes; run after run_experiments.py

Writes to results/ and figures/:
  model_details.csv          what each trained model looks like inside, and how fast it is
  example_predictions.csv    the trained model making predictions on recordings it never saw
  roc_seizure_vs_all.csv     the trade-off between catching seizures and false alarms
  learning_curve.csv         how accuracy changes with more training recordings
  figures 08-12              training vs inference, brain-wave bands, wave size, ROC, learning curve
"""

import json
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from sklearn.base import clone
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

import make_figures as mf  # shared colours and chart style
from eeg.data import SET_ORDER, bandpass, load_bonn, make_windows
from eeg.experiments import make_models, task_data
from eeg.features import FEATURE_NAMES, extract_features

warnings.filterwarnings("ignore", category=UserWarning)
RES, FIG = Path("results"), Path("figures")


def count_learned_numbers(name, model):
    """Roughly how many numbers the model learned during training, in plain terms."""
    est = model.steps[-1][1] if hasattr(model, "steps") else model
    if name.startswith("Logistic"):
        n = est.coef_.size + est.intercept_.size
        return n, f"{n} numbers (one weight per feature + 1)"
    if name.startswith("SVM"):
        sv = int(est.n_support_.sum())
        return sv * (len(FEATURE_NAMES) + 1) + 1, f"kept {sv:,} training windows as 'support vectors'"
    if name.startswith("Random"):
        nodes = sum(t.tree_.node_count for t in est.estimators_)
        return nodes, f"{len(est.estimators_)} trees, {nodes:,} yes/no nodes in total"
    if name.startswith("Gradient"):
        try:
            nodes = sum(p[0].nodes.shape[0] for p in est._predictors)
            return nodes, f"{est.n_iter_} trees, {nodes:,} yes/no nodes in total"
        except Exception:
            return np.nan, f"{est.n_iter_} trees"
    if name.startswith("Neural"):
        n = sum(c.size for c in est.coefs_) + sum(b.size for b in est.intercepts_)
        layers = " → ".join(str(s) for s in [len(FEATURE_NAMES), *est.hidden_layer_sizes, est.n_outputs_])
        return n, f"{n:,} connection strengths (layers {layers})"
    return np.nan, ""


SETTINGS = {
    "Logistic regression": "features rescaled; groups balanced",
    "SVM (RBF kernel)": "curved (RBF) boundary; C = 10; groups balanced",
    "Random forest": "300 trees; groups balanced",
    "Gradient boosting": "up to 100 trees added one at a time; learning rate 0.1",
    "Neural network (MLP)": "2 hidden layers (64, 32); stops early when it stops improving",
}


def model_details(X, window_sets, groups):
    Xt, y, g, _ = task_data("seizure_vs_all", X, window_sets, groups)
    rows = []
    for name, m in make_models().items():
        m = clone(m)
        t0 = time.perf_counter()
        m.fit(Xt, y)
        fit_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        for _ in range(5):
            m.predict(Xt)
        per_window_ms = (time.perf_counter() - t0) / 5 / len(Xt) * 1000
        n, desc = count_learned_numbers(name, m)
        rows.append({"model": name, "settings": SETTINGS[name], "what_it_learned": desc,
                     "learned_numbers": n, "training_seconds": round(fit_s, 2),
                     "prediction_ms_per_window": round(per_window_ms, 4)})
    df = pd.DataFrame(rows)
    df.to_csv(RES / "model_details.csv", index=False)
    return df


def example_predictions(X, window_sets, groups):
    """Train once on 80% of recordings, then predict ('inference') on the unseen 20%."""
    Xt, y, g, sets = task_data("seizure_vs_all", X, window_sets, groups)
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=0).split(Xt, y, g))
    m = clone(make_models()["Gradient boosting"]).fit(Xt[tr], y[tr])
    prob = m.predict_proba(Xt[te])[:, 1]
    pred = (prob >= 0.5).astype(int)
    df = pd.DataFrame({"recording": g[te], "group": sets[te], "true": y[te], "pred": pred, "prob": prob})
    per_rec = df.groupby(["recording", "group"]).agg(
        true=("true", "first"), windows=("pred", "size"), windows_flagged=("pred", "sum"),
        mean_seizure_probability=("prob", "mean")).reset_index()
    per_rec["verdict"] = np.where(per_rec.windows_flagged > per_rec.windows / 2, "Seizure", "No seizure")
    per_rec["correct"] = (per_rec.verdict == "Seizure") == (per_rec.true == 1)
    per_rec.to_csv(RES / "example_predictions.csv", index=False)
    summary = {
        "train_recordings": int(len(set(g[tr]))), "test_recordings": int(len(set(g[te]))),
        "test_windows": int(len(te)),
        "window_balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
        "recordings_correct": int(per_rec.correct.sum()),
        "per_group": per_rec.groupby("group").agg(n=("correct", "size"), correct=("correct", "sum")).reset_index().to_dict("records"),
    }
    (RES / "example_predictions_summary.json").write_text(json.dumps(summary, indent=1))
    return per_rec, summary


def roc_seizure(X, window_sets, groups):
    Xt, y, g, _ = task_data("seizure_vs_all", X, window_sets, groups)
    prob = np.zeros(len(y))
    for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0).split(Xt, y, g):
        prob[te] = clone(make_models()["Gradient boosting"]).fit(Xt[tr], y[tr]).predict_proba(Xt[te])[:, 1]
    fpr, tpr, thr = roc_curve(y, prob)
    pd.DataFrame({"false_alarm_rate": fpr, "seizures_caught": tpr, "threshold": thr}).to_csv(RES / "roc_seizure_vs_all.csv", index=False)
    pred = prob >= 0.5
    sens = pred[y == 1].mean()
    spec = 1 - pred[y == 0].mean()
    return fpr, tpr, roc_auc_score(y, prob), sens, spec


def learning_curve(X, window_sets, groups, fractions=(0.1, 0.2, 0.4, 0.6, 0.8, 1.0), seeds=(0, 1, 2)):
    rows = []
    for task in ("seizure_vs_all", "zone_vs_opposite"):
        Xt, y, g, _ = task_data(task, X, window_sets, groups)
        for seed in seeds:
            rng = np.random.default_rng(seed)
            for tr, te in StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed).split(Xt, y, g):
                recs = np.unique(g[tr])
                rec_label = {r: y[tr][g[tr] == r][0] for r in recs}
                for f in fractions:
                    keep = []
                    for lab in (0, 1):  # keep the same mix of groups at every size
                        pool = np.array([r for r in recs if rec_label[r] == lab])
                        k = max(2, int(round(f * len(pool))))
                        keep.extend(rng.choice(pool, size=min(k, len(pool)), replace=False))
                    sub = tr[np.isin(g[tr], keep)]
                    m = clone(make_models()["Gradient boosting"]).set_params(random_state=seed).fit(Xt[sub], y[sub])
                    rows.append({"task": task, "fraction": f, "train_recordings": len(keep),
                                 "balanced_accuracy": balanced_accuracy_score(y[te], m.predict(Xt[te]))})
    df = pd.DataFrame(rows)
    out = df.groupby(["task", "fraction"]).agg(train_recordings=("train_recordings", "mean"),
                                               balanced_accuracy=("balanced_accuracy", "mean"),
                                               std=("balanced_accuracy", "std")).reset_index()
    out.to_csv(RES / "learning_curve.csv", index=False)
    return out


# ------------------------------------------------------------------ figures
def fig_training_vs_inference():
    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, sub, color):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                    fc=mf.SURFACE, ec=color, lw=2))
        ax.text(x + w / 2, y + h * 0.64, title, ha="center", va="center", fontsize=11, fontweight="bold", color=mf.INK)
        ax.text(x + w / 2, y + h * 0.3, sub, ha="center", va="center", fontsize=8.8, color=mf.INK2)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, color=mf.MUTED, lw=1.4))

    ax.text(0.05, 4.35, "TRAINING  (learning from examples with known answers)", fontsize=10.5, fontweight="bold", color=mf.SERIES[0])
    steps = [("EEG recordings", "+ the right answers"), ("Clean", "filter 0.5–40 Hz"), ("Cut", "2-second windows"),
             ("Measure", "18 features each"), ("Fit the model", "adjust to cut mistakes")]
    xs = [0.05, 2.25, 4.45, 6.65, 8.85]
    for (t, s), x in zip(steps, xs):
        box(x, 2.85, 2.0, 1.15, t, s, mf.SERIES[0])
    for x in xs[:-1]:
        arrow(x + 2.0, 3.42, x + 2.25, 3.42)

    ax.text(0.05, 1.95, "INFERENCE  (using the trained model on new data)", fontsize=10.5, fontweight="bold", color=mf.SERIES[1])
    steps2 = [("New recording", "no answer given"), ("Clean", "same filter"), ("Cut", "same windows"),
              ("Measure", "same 18 features"), ("Trained model", "answers: seizure or not")]
    for (t, s), x in zip(steps2, xs):
        box(x, 0.45, 2.0, 1.15, t, s, mf.SERIES[1])
    for x in xs[:-1]:
        arrow(x + 2.0, 1.02, x + 2.25, 1.02)
    arrow(9.85, 2.85, 9.85, 1.6)
    ax.text(9.95, 2.22, "the learned\nrules are reused", fontsize=8.5, color=mf.INK2, va="center")
    mf.save(fig, "08_training_vs_inference.png")


def fig_brain_wave_bands():
    bands = [("Delta", "0.5–4 Hz", 2, "deep sleep"), ("Theta", "4–8 Hz", 6, "drowsiness, light sleep"),
             ("Alpha", "8–13 Hz", 10, "relaxed and awake, eyes closed"), ("Beta", "13–30 Hz", 20, "alert, thinking, focused"),
             ("Gamma", "30+ Hz", 35, "busy processing")]
    t = np.linspace(0, 1, 2000)
    fig, axes = plt.subplots(len(bands), 1, figsize=(10, 5.6), sharex=True)
    for ax, (name, rng, f, state) in zip(axes, bands):
        ax.plot(t, np.sin(2 * np.pi * f * t), color=mf.SERIES[0], lw=1.2)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        ax.text(-0.02, 0.5, f"{name}\n{rng}", transform=ax.transAxes, ha="right", va="center", fontsize=10,
                fontweight="bold", color=mf.INK)
        ax.text(1.02, 0.5, state, transform=ax.transAxes, ha="left", va="center", fontsize=9.5, color=mf.INK2)
    axes[-1].set_xlabel("One second of time")
    axes[0].set_title("Brain-wave rhythms: slower waves are typical of sleep, faster waves of alert activity\n"
                      "(simplified sine waves for illustration, not real EEG)", fontsize=11, pad=10)
    fig.subplots_adjust(left=0.13, right=0.78, hspace=0.35)
    mf.save(fig, "09_brain_wave_bands.png")


def fig_wave_size(X, window_sets):
    size = np.exp(X[:, FEATURE_NAMES.index("log_std")])
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.grid(True, axis="y", which="major")
    rng = np.random.default_rng(0)
    for i, s in enumerate(SET_ORDER):
        v = size[window_sets == s]
        ax.scatter(i + rng.uniform(-0.28, 0.28, len(v)), v, s=6, alpha=0.35, color=mf.SET_COLOR[s], lw=0, zorder=2)
        ax.plot([i - 0.32, i + 0.32], [np.median(v)] * 2, color=mf.INK, lw=2, zorder=3)
        ax.text(i + 0.36, np.median(v), f"{np.median(v):.0f} µV", va="center", fontsize=9, color=mf.INK)
    ax.set_yscale("log")
    ax.set_xticks(range(5), ["A · Healthy\neyes open", "B · Healthy\neyes closed", "C · Epilepsy\nopposite side", "D · Epilepsy\nseizure zone", "E · Epilepsy\nduring seizure"])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Wave size: standard deviation (µV, log scale)")
    ax.set_title("Wave size of every 2-second window, by group (black line = median)", pad=12)
    mf.save(fig, "10_wave_size_by_group.png")


def fig_roc(fpr, tpr, auc, sens, spec):
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    ax.grid(True)
    ax.plot([0, 1], [0, 1], color=mf.AXIS, lw=1.2, ls="--", zorder=1)
    ax.text(0.8, 0.73, "coin flip", rotation=38, color=mf.MUTED, fontsize=9)
    ax.plot(fpr, tpr, color=mf.SERIES[0], lw=2.2, zorder=2, label=f"Gradient boosting (AUC = {auc:.3f})")
    ax.scatter([1 - spec], [sens], s=70, color=mf.SERIES[1], edgecolor=mf.SURFACE, lw=2, zorder=3,
               label=f"Default setting: catches {sens*100:.1f}%, false alarms {(1-spec)*100:.1f}%")
    ax.set_xlim(-0.01, 1)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("False alarm rate (non-seizure windows flagged)")
    ax.set_ylabel("Seizures caught (sensitivity)")
    ax.set_title("ROC curve: seizure vs all non-seizure EEG\n(windows from held-out recordings)", fontsize=11.5)
    ax.legend(loc="lower right", fontsize=8.8)
    inset = ax.inset_axes([0.42, 0.3, 0.5, 0.36])
    inset.plot(fpr, tpr, color=mf.SERIES[0], lw=2)
    inset.scatter([1 - spec], [sens], s=40, color=mf.SERIES[1], edgecolor=mf.SURFACE, lw=1.5, zorder=3)
    inset.set_xlim(0, 0.1)
    inset.set_ylim(0.8, 1.0)
    inset.set_title("zoomed in", fontsize=8.5, color=mf.INK2)
    inset.tick_params(labelsize=8)
    mf.save(fig, "11_roc_seizure_detection.png")


def fig_learning_curve(lc):
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.grid(True, axis="y")
    labels = {"seizure_vs_all": "Seizure vs all non-seizure", "zone_vs_opposite": "Seizure zone vs opposite side"}
    for k, (task, lab) in enumerate(labels.items()):
        d = lc[lc.task == task]
        ax.plot(d.train_recordings, d.balanced_accuracy * 100, marker="o", ms=7, lw=2, color=mf.SERIES[k],
                label=lab, markeredgecolor=mf.SURFACE, markeredgewidth=1.5)
        last = d.iloc[-1]
        ax.text(last.train_recordings + 6, last.balanced_accuracy * 100, f"{last.balanced_accuracy*100:.1f}%",
                va="center", fontsize=9, color=mf.INK)
    ax.axhline(50, color=mf.AXIS, ls="--", lw=1)
    ax.text(5, 51, "coin flip", fontsize=9, color=mf.MUTED)
    ax.set_ylim(45, 102)
    ax.set_xlabel("Number of recordings the model trained on")
    ax.set_ylabel("Balanced accuracy on unseen recordings (%)")
    ax.set_title("Does more training data help? (gradient boosting)", pad=26)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, fontsize=9)
    mf.save(fig, "12_learning_curve.png")


def main(data_dir="data/bonn"):
    t0 = time.time()
    signals, set_labels, rec_ids = load_bonn(data_dir)
    windows, window_sets, groups = make_windows(bandpass(signals), set_labels, rec_ids)
    X = extract_features(windows)

    md = model_details(X, window_sets, groups)
    print(md.to_string(index=False))
    per_rec, summary = example_predictions(X, window_sets, groups)
    print(json.dumps(summary, indent=1))
    fpr, tpr, auc, sens, spec = roc_seizure(X, window_sets, groups)
    print(f"ROC AUC {auc:.4f}; at 0.5: sensitivity {sens:.4f}, specificity {spec:.4f}")
    lc = learning_curve(X, window_sets, groups)
    print(lc.to_string(index=False))
    (RES / "roc_summary.json").write_text(json.dumps({"auc": auc, "sensitivity": sens, "specificity": spec}, indent=1))

    fig_training_vs_inference()
    fig_brain_wave_bands()
    fig_wave_size(X, window_sets)
    fig_roc(fpr, tpr, auc, sens, spec)
    fig_learning_curve(lc)
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()

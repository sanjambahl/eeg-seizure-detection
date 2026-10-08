"""Draw every chart in figures/ from the files in results/.

Run after run_experiments.py:
    python make_figures.py
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import welch

from eeg.data import FS, SET_ORDER, SETS, bandpass, load_bonn
from eeg.experiments import TASK_LABELS, TASKS
from eeg.features import FEATURE_LABELS

# ---- palette (validated for colour-blind separation) ----
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SET_COLOR = dict(zip(SET_ORDER, SERIES))
BLUES = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5",
         "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SET_SHORT = {
    "Z": "A · Healthy, eyes open",
    "O": "B · Healthy, eyes closed",
    "N": "C · Epilepsy, opposite side",
    "F": "D · Epilepsy, seizure zone",
    "S": "E · Epilepsy, during seizure",
}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.family": "sans-serif", "font.size": 10, "text.color": INK,
    "axes.labelcolor": INK2, "axes.edgecolor": AXIS, "axes.titlecolor": INK,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.8,
    "legend.frameon": False, "legend.labelcolor": INK2,
})

RES, FIG = Path("results"), Path("figures")
MODEL_ORDER = ["Logistic regression", "SVM (RBF kernel)", "Random forest",
               "Gradient boosting", "Neural network (MLP)"]


def blues_cmap():
    from matplotlib.colors import LinearSegmentedColormap
    return LinearSegmentedColormap.from_list("blues", BLUES)


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("  wrote", FIG / name)


def annotate_heatmap(ax, values, fmt, vmin, vmax):
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            v = values[i, j]
            frac = (v - vmin) / (vmax - vmin + 1e-12)
            ax.text(j, i, fmt(v), ha="center", va="center", fontsize=9.5,
                    color="#ffffff" if frac > 0.55 else INK)


def fig_example_signals(data_dir):
    signals, sets, _ = load_bonn(data_dir)
    x = bandpass(signals)
    t = np.arange(int(8 * FS)) / FS
    first = {s: x[np.where(sets == s)[0][0], : len(t)] for s in SET_ORDER}
    lim = np.ceil(max(np.abs(v).max() for v in first.values()) / 500) * 500
    fig, axes = plt.subplots(5, 1, figsize=(10, 7.5), sharex=True, sharey=True)
    for ax, s in zip(axes, SET_ORDER):
        rec = first[s]
        ax.plot(t, rec, color=SET_COLOR[s], lw=1.0)
        ax.text(0.0, 1.02, SET_SHORT[s], transform=ax.transAxes, fontsize=10,
                color=INK, fontweight="bold", va="bottom")
        ax.set_ylim(-lim, lim)
        ax.axhline(0, color=GRID, lw=0.8, zorder=0)
    axes[2].set_ylabel("Voltage (µV)")
    axes[-1].set_xlabel("Time (seconds)")
    fig.suptitle("What the five groups look like (first 8 s of one recording each, same scale)",
                 x=0.06, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(h_pad=1.4)
    save(fig, "01_example_signals.png")


def fig_power_spectra(data_dir):
    signals, sets, _ = load_bonn(data_dir)
    x = bandpass(signals)
    f, p = welch(x, fs=FS, nperseg=512, axis=1)
    keep = (f >= 0.5) & (f <= 40)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.grid(True, axis="y", which="major")
    for s in SET_ORDER:
        mean_log = np.log10(p[sets == s][:, keep]).mean(axis=0)
        ax.plot(f[keep], 10 ** mean_log, color=SET_COLOR[s], lw=2, label=SET_SHORT[s])
    ax.set_yscale("log")
    ax.set_xlim(0, 40)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power (µV² / Hz, log scale)")
    for name, (lo, hi) in {"δ": (0.5, 4), "θ": (4, 8), "α": (8, 13), "β": (13, 30), "γ": (30, 40)}.items():
        ax.text((lo + hi) / 2, 1.01, name, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom", color=MUTED, fontsize=10)
        ax.axvline(hi, color=GRID, lw=0.8, zorder=0)
    ax.set_title("Average power spectrum of each group", pad=22)
    ax.legend(loc="upper right", fontsize=9)
    save(fig, "02_power_spectra.png")


def fig_model_heatmap(cv):
    honest = cv[cv.split == "grouped"]
    tasks = list(TASKS)
    vals = np.array([[honest[(honest.task == t) & (honest.model == m)].balanced_accuracy_mean.iloc[0]
                      for m in MODEL_ORDER] for t in tasks]) * 100
    fig, ax = plt.subplots(figsize=(10, 4.6))
    vmin, vmax = 60, 100
    ax.imshow(vals, cmap=blues_cmap(), vmin=vmin, vmax=vmax, aspect="auto")
    annotate_heatmap(ax, vals, lambda v: f"{v:.1f}%", vmin, vmax)
    ax.set_xticks(range(len(MODEL_ORDER)), [m.replace(" (", "\n(") for m in MODEL_ORDER])
    ax.set_yticks(range(len(tasks)), [TASK_LABELS[t] for t in tasks])
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, len(MODEL_ORDER)), minor=True)
    ax.set_yticks(np.arange(-.5, len(tasks)), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title("Balanced accuracy on recordings the model never saw\n"
                 "(mean of 3 × 5-fold cross-validation; 50% = coin flip)", fontsize=11.5)
    save(fig, "03_model_comparison.png")


def fig_leakage(cv):
    tasks = list(TASKS)
    honest = cv[cv.split == "grouped"].groupby("task").balanced_accuracy_mean.mean().reindex(tasks) * 100
    leaky = cv[cv.split == "leaky"].groupby("task").balanced_accuracy_mean.mean().reindex(tasks) * 100
    fig, ax = plt.subplots(figsize=(9, 4.6))
    y = np.arange(len(tasks))[::-1]
    ax.grid(True, axis="x")
    for yi, h, l in zip(y, honest, leaky):
        ax.plot([h, l], [yi, yi], color=AXIS, lw=2, zorder=1)
        ax.text(max(h, l) + 0.8, yi, f"+{l - h:.1f} pts", va="center", fontsize=9, color=INK2)
    ax.scatter(honest, y, s=70, color=SERIES[0], zorder=3, edgecolor=SURFACE, linewidth=2,
               label="Honest: whole recordings held out")
    ax.scatter(leaky, y, s=90, facecolor="none", edgecolor=SERIES[1], zorder=4, linewidth=2.2,
               label="Leaky: windows shuffled, recordings split across train/test")
    ax.set_yticks(y, [TASK_LABELS[t] for t in tasks])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlim(65, 103)
    ax.set_xlabel("Balanced accuracy (%), averaged over the five models")
    ax.set_title("Shuffling windows inflates accuracy, most on the hardest tasks", pad=34)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), ncol=1, fontsize=9, handletextpad=0.3)
    save(fig, "04_leakage.png")


def fig_transfer(tr):
    dets = list(tr.detector.drop_duplicates())
    groups = ["Z", "O", "N", "F"]
    agg = tr.groupby(["detector", "test_set"]).predicted_seizure_rate_mean.agg(["mean", "min", "max"])
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.grid(True, axis="y")
    width = 0.26
    handles = []
    for k, d in enumerate(dets):
        xs = np.arange(len(groups)) + (k - 1) * (width + 0.02)
        m = np.array([agg.loc[(d, g), "mean"] for g in groups]) * 100
        lo = m - np.array([agg.loc[(d, g), "min"] for g in groups]) * 100
        hi = np.array([agg.loc[(d, g), "max"] for g in groups]) * 100 - m
        handles.append(ax.bar(xs, m, width, color=SERIES[k], zorder=2))
        ax.errorbar(xs, m, yerr=[lo, hi], fmt="none", ecolor=INK2, elinewidth=1, capsize=2, zorder=3)
        for xi, v in zip(xs, m):
            ax.text(xi, v + hi[list(xs).index(xi)] + 0.8, f"{v:.0f}%", ha="center", fontsize=8.5, color=INK2)
    ax.set_xticks(np.arange(len(groups)), [SET_SHORT[g].replace(" · ", "\n") for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Non-seizure windows flagged as seizure (%)")
    sens = {d: agg.loc[(d, "S"), "mean"] * 100 for d in dets}
    ax.set_title("False alarms on held-out non-seizure EEG", pad=64)
    ax.legend(handles, [f"{d}  (catches {sens[d]:.1f}% of seizure windows)" for d in dets],
              loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=9)
    ax.text(1.0, -0.2, "Bars: mean over five models · whiskers: lowest–highest model",
            transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
    save(fig, "05_false_alarms.png")


def fig_importance(imp):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    titles = {"seizure_vs_interictal": "Seizure vs same patients, no seizure",
              "zone_vs_opposite": "Seizure zone vs opposite side"}
    for ax, (task, title) in zip(axes, titles.items()):
        d = imp[imp.task == task].sort_values("mean", ascending=False).head(8)[::-1]
        ax.grid(True, axis="x")
        ax.barh([FEATURE_LABELS[f] for f in d.feature], d["mean"] * 100, color=SERIES[0],
                height=0.62, zorder=2)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Accuracy lost when this feature is scrambled (pts)")
    fig.suptitle("What the gradient-boosting model relies on (permutation importance, top 8)",
                 x=0.02, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout(w_pad=3)
    save(fig, "06_feature_importance.png")


def fig_confusion(conf, model):
    cm = np.array(conf["matrices"][model], dtype=float)
    pct = cm / cm.sum(axis=1, keepdims=True) * 100
    labels = [SET_SHORT[s] for s in conf["labels"]]
    fig, ax = plt.subplots(figsize=(7.8, 6.2))
    ax.imshow(pct, cmap=blues_cmap(), vmin=0, vmax=100)
    annotate_heatmap(ax, pct, lambda v: f"{v:.0f}%", 0, 100)
    ax.set_xticks(range(5), [l.split(" · ")[0] for l in labels])
    ax.set_yticks(range(5), labels)
    ax.set_xlabel("Predicted group")
    ax.set_ylabel("True group")
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_xticks(np.arange(-.5, 5), minor=True)
    ax.set_yticks(np.arange(-.5, 5), minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=2)
    ax.tick_params(which="minor", length=0)
    ax.set_title(f"Five-group confusion matrix ({model})\nEach row: where that group's windows ended up",
                 fontsize=11.5)
    save(fig, "07_confusion_five_class.png")


def main(data_dir="data/bonn"):
    FIG.mkdir(exist_ok=True)
    cv = pd.read_csv(RES / "cv_results.csv")
    fig_example_signals(data_dir)
    fig_power_spectra(data_dir)
    fig_model_heatmap(cv)
    fig_leakage(cv)
    fig_transfer(pd.read_csv(RES / "detector_transfer.csv"))
    fig_importance(pd.read_csv(RES / "feature_importance.csv"))
    five = cv[(cv.task == "five_class") & (cv.split == "grouped")]
    best = five.sort_values("balanced_accuracy_mean").model.iloc[-1]
    fig_confusion(json.loads((RES / "confusion_five_class.json").read_text()), best)


if __name__ == "__main__":
    main()

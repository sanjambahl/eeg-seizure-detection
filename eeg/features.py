"""Turning each EEG window into a short list of numbers (features) a model can learn from.

Features fall into two groups:

* AMPLITUDE features depend on how big the signal is (in microvolts). Seizures
  are often huge, so these are powerful, but they also change with the type of
  electrode and where it sits, which makes them a possible shortcut.
* SHAPE features ignore overall size: they describe rhythm and complexity
  (which frequencies dominate, how irregular the wave is). Multiplying a signal
  by 10 leaves every shape feature unchanged.
"""

import math

import numpy as np
from scipy.signal import welch
from scipy.stats import kurtosis, skew

from .data import FS

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 40),
}

AMPLITUDE_FEATURES = ["log_std", "log_line_length", "log_range", "log_total_power"]
SHAPE_FEATURES = [
    "rel_delta", "rel_theta", "rel_alpha", "rel_beta", "rel_gamma",
    "peak_freq", "spectral_edge_90", "spectral_entropy",
    "hjorth_mobility", "hjorth_complexity",
    "zero_cross_rate", "skewness", "kurtosis", "perm_entropy",
]
FEATURE_NAMES = AMPLITUDE_FEATURES + SHAPE_FEATURES

# Plain-English labels for charts and the README
FEATURE_LABELS = {
    "log_std": "Amplitude (std)",
    "log_line_length": "Line length",
    "log_range": "Peak-to-peak range",
    "log_total_power": "Total power",
    "rel_delta": "Delta share (0.5-4 Hz)",
    "rel_theta": "Theta share (4-8 Hz)",
    "rel_alpha": "Alpha share (8-13 Hz)",
    "rel_beta": "Beta share (13-30 Hz)",
    "rel_gamma": "Gamma share (30-40 Hz)",
    "peak_freq": "Peak frequency",
    "spectral_edge_90": "Spectral edge (90%)",
    "spectral_entropy": "Spectral entropy",
    "hjorth_mobility": "Hjorth mobility",
    "hjorth_complexity": "Hjorth complexity",
    "zero_cross_rate": "Zero-crossing rate",
    "skewness": "Skewness",
    "kurtosis": "Kurtosis",
    "perm_entropy": "Permutation entropy",
}


def permutation_entropy(windows, order=3, delay=1):
    """How unpredictable the up/down pattern of the signal is (0 = perfectly regular, 1 = random).

    Looks at every run of `order` consecutive samples, records their ranking
    pattern (e.g. low-high-mid), and measures how evenly the 6 possible patterns occur.
    """
    n = windows.shape[1] - (order - 1) * delay
    stacked = np.stack([windows[:, i * delay: i * delay + n] for i in range(order)], axis=-1)
    patterns = np.argsort(stacked, axis=-1)
    codes = (patterns * (order ** np.arange(order))).sum(-1)  # one integer per pattern
    n_codes = order ** order
    counts = np.stack([(codes == c).sum(1) for c in range(n_codes)], axis=1)
    p = counts / counts.sum(1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -np.nansum(np.where(p > 0, p * np.log(p), 0.0), axis=1)
    return h / np.log(math.factorial(order))


def extract_features(windows, fs=FS):
    """Compute all features for every window. Returns an array of shape (n_windows, n_features)."""
    x = windows - windows.mean(axis=1, keepdims=True)
    dx = np.diff(x, axis=1)
    ddx = np.diff(dx, axis=1)

    # --- amplitude features ---
    std = x.std(axis=1)
    line_length = np.abs(dx).mean(axis=1)
    rng = x.max(axis=1) - x.min(axis=1)

    # --- frequency content (power spectrum via Welch's method) ---
    nperseg = min(256, x.shape[1])
    freqs, psd = welch(x, fs=fs, nperseg=nperseg, axis=1)
    in_range = (freqs >= 0.5) & (freqs <= 40)
    f, p = freqs[in_range], psd[:, in_range]
    total = p.sum(axis=1)
    rel = {name: p[:, (f >= lo) & (f < hi)].sum(axis=1) / total for name, (lo, hi) in BANDS.items()}
    peak_freq = f[np.argmax(p, axis=1)]
    cum = np.cumsum(p, axis=1) / total[:, None]
    edge90 = f[np.argmax(cum >= 0.9, axis=1)]
    pn = p / total[:, None]
    spec_entropy = -(pn * np.log(pn + 1e-12)).sum(axis=1) / np.log(pn.shape[1])

    # --- Hjorth parameters: how fast (mobility) and how irregular (complexity) the wave is ---
    var_x, var_dx, var_ddx = x.var(axis=1), dx.var(axis=1), ddx.var(axis=1)
    mobility = np.sqrt(var_dx / var_x)
    complexity = np.sqrt(var_ddx / var_dx) / mobility

    zero_cross = (np.diff(np.sign(x), axis=1) != 0).mean(axis=1)

    feats = {
        "log_std": np.log(std),
        "log_line_length": np.log(line_length),
        "log_range": np.log(rng),
        "log_total_power": np.log(total),
        "rel_delta": rel["delta"],
        "rel_theta": rel["theta"],
        "rel_alpha": rel["alpha"],
        "rel_beta": rel["beta"],
        "rel_gamma": rel["gamma"],
        "peak_freq": peak_freq,
        "spectral_edge_90": edge90,
        "spectral_entropy": spec_entropy,
        "hjorth_mobility": mobility,
        "hjorth_complexity": complexity,
        "zero_cross_rate": zero_cross,
        "skewness": skew(x, axis=1),
        "kurtosis": kurtosis(x, axis=1),
        "perm_entropy": permutation_entropy(x),
    }
    return np.column_stack([feats[name] for name in FEATURE_NAMES])

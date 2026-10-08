"""Train and test the five models on your own EEG recordings.

    python examples/use_your_own_data.py my_data 256

my_data = a folder with one sub-folder per label (for example seizure/ and normal/),
          each holding one recording per file (see eeg/custom.py)
256     = sampling rate of your recordings, in samples per second
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eeg.custom import prepare  # noqa: E402
from eeg.experiments import cross_validate, make_models  # noqa: E402

folder = sys.argv[1] if len(sys.argv) > 1 else "my_data"
fs = float(sys.argv[2]) if len(sys.argv) > 2 else 256.0

X, y, groups, classes = prepare(folder, fs=fs)
print(f"{len(set(groups))} recordings -> {len(y)} windows; labels: {classes}; windows per label: {np.bincount(y).tolist()}")

n_splits = min(5, min(len(set(groups[y == k])) for k in range(len(classes))))
if n_splits < 2:
    sys.exit("Each label needs at least 2 recordings so some can be held out for testing.")

rows = []
for name, model in make_models().items():
    r = cross_validate(model, X, y, groups, split="grouped", n_splits=n_splits, seeds=(0, 1, 2))
    rows.append({"model": name,
                 "balanced accuracy per window (%)": 100 * np.mean(r["balanced_accuracy"]),
                 "balanced accuracy per recording (%)": 100 * np.mean(r["recording_balanced_accuracy"])})
print(pd.DataFrame(rows).round(1).to_string(index=False))

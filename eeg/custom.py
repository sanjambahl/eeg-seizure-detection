"""Use the same pipeline on your own EEG recordings.

Put one recording per file, sorted into one folder per answer (label):

    my_data/
        seizure/     rec01.txt   rec02.csv   ...
        normal/      rec03.txt   rec04.txt   ...

Each file holds one EEG channel: one number per line (.txt), or a column in a .csv.
The folder names become the answers the model learns. Recordings can have different lengths.

    from eeg.custom import prepare
    X, y, groups, classes = prepare("my_data", fs=256)   # fs = samples per second of YOUR recordings
"""

from pathlib import Path

import numpy as np

from .data import bandpass
from .features import extract_features

SUFFIXES = {".txt", ".csv", ".tsv"}


def read_recording(path, channel=0):
    """Read one single-channel recording from a .txt, .csv or .tsv file."""
    path = Path(path)
    delimiter = "," if path.suffix.lower() == ".csv" else ("\t" if path.suffix.lower() == ".tsv" else None)
    data = np.genfromtxt(path, delimiter=delimiter)
    if data.ndim == 2:
        if np.isnan(data[0]).all():  # first row was a text header
            data = data[1:]
        data = data[:, channel]
    data = data[~np.isnan(data)]
    if data.size == 0:
        raise ValueError(f"No numbers found in {path}")
    return data.astype(float)


def load_folder(folder, channel=0):
    """Load every recording in folder/<label>/<file>.

    Returns (signals, labels, recording_ids): a list of 1-D arrays, and matching label and id arrays.
    """
    folder = Path(folder)
    signals, labels, ids = [], [], []
    for label_dir in sorted(p for p in folder.iterdir() if p.is_dir()):
        for f in sorted(label_dir.iterdir()):
            if f.suffix.lower() in SUFFIXES:
                signals.append(read_recording(f, channel))
                labels.append(label_dir.name)
                ids.append(f"{label_dir.name}/{f.stem}")
    if len(set(labels)) < 2:
        raise ValueError(f"Need at least two label folders with recordings inside {folder}")
    return signals, np.array(labels), np.array(ids)


def prepare(folder, fs, window_sec=2.0, low=0.5, high=40.0, channel=0):
    """Clean, cut into windows and measure the 18 features, exactly as for the Bonn data.

    Returns
    -------
    X : features, one row per window
    y : answer for each window, as a number (0, 1, ...)
    groups : which recording each window came from (keeps recordings together when testing)
    classes : the label names, in the order of the numbers in y
    """
    high = min(high, 0.45 * fs)  # the filter's top frequency must be below half the sampling rate
    n = int(round(window_sec * fs))
    signals, labels, ids = load_folder(folder, channel)
    classes = sorted(set(str(l) for l in labels))
    windows, y, groups = [], [], []
    skipped = 0
    for sig, lab, rid in zip(signals, labels, ids):
        if len(sig) < n:
            skipped += 1
            continue
        clean = bandpass(sig[None, :], low=low, high=high, fs=fs)[0]
        k = len(clean) // n
        windows.append(clean[: k * n].reshape(k, n))
        y += [classes.index(str(lab))] * k
        groups += [rid] * k
    if skipped:
        print(f"Skipped {skipped} recording(s) shorter than one {window_sec}-second window.")
    X = extract_features(np.vstack(windows), fs=fs)
    return X, np.array(y), np.array(groups), classes

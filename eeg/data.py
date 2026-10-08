"""Loading and preparing the Bonn EEG dataset.

The Bonn dataset (Andrzejak et al., 2001) has five sets of 100 single-channel
EEG recordings each. Every recording is 23.6 seconds long, sampled at 173.61 Hz
(4097 samples).

    Set  Folder  Who / where                                   During a seizure?
    ---  ------  --------------------------------------------  -----------------
    A    Z       Healthy volunteers, scalp, eyes open          no
    B    O       Healthy volunteers, scalp, eyes closed        no
    C    N       Epilepsy patients, inside the skull,          no
                 hippocampus on the side OPPOSITE the focus
    D    F       Epilepsy patients, inside the skull,          no
                 INSIDE the zone where seizures start
    E    S       Epilepsy patients, inside the skull           YES

Note the catch: sets A and B were recorded on the scalp, but C, D and E were
recorded with electrodes inside the skull. Anything that separates "healthy"
from "epilepsy" in this dataset might be picking up that difference instead of
the disease.
"""

from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt

FS = 173.61  # sampling rate in Hz

# Folder letter -> (paper set name, short human-readable description)
SETS = {
    "Z": ("A", "Healthy, eyes open (scalp)"),
    "O": ("B", "Healthy, eyes closed (scalp)"),
    "N": ("C", "Epilepsy, between seizures, opposite side (intracranial)"),
    "F": ("D", "Epilepsy, between seizures, seizure zone (intracranial)"),
    "S": ("E", "Epilepsy, during a seizure (intracranial)"),
}
SET_ORDER = ["Z", "O", "N", "F", "S"]


def load_bonn(data_dir):
    """Load every recording.

    Returns
    -------
    signals : float array, shape (500, 4097)
    set_labels : array of folder letters ("Z", "O", "N", "F", "S"), one per recording
    recording_ids : array of file names like "Z001", one per recording
    """
    data_dir = Path(data_dir)
    all_txt = [p for p in data_dir.rglob("*") if p.suffix.lower() == ".txt"]
    signals, set_labels, recording_ids = [], [], []
    for letter in SET_ORDER:
        # Files are named like Z001.txt (set N uses upper-case N001.TXT), so match case-insensitively
        files = sorted(
            (p for p in all_txt if p.stem[:1].upper() == letter and p.stem[1:].isdigit()),
            key=lambda p: p.stem.upper(),
        )
        if len(files) != 100:
            raise FileNotFoundError(
                f"Expected 100 files for set {letter} under {data_dir}, found {len(files)}. "
                "Run `python scripts/download_data.py` first."
            )
        for f in files:
            x = np.loadtxt(f)
            signals.append(x)
            set_labels.append(letter)
            recording_ids.append(f.stem.upper())
    signals = np.vstack(signals)
    return signals, np.array(set_labels), np.array(recording_ids)


def bandpass(signals, low=0.5, high=40.0, fs=FS, order=4):
    """Zero-phase band-pass filter: keeps 0.5-40 Hz, removes slow drift and high-frequency noise."""
    sos = butter(order, [low, high], btype="band", fs=fs, output="sos")
    return sosfiltfilt(sos, signals, axis=-1)


def make_windows(signals, set_labels, recording_ids, window_sec=2.0, fs=FS):
    """Cut every 23.6 s recording into non-overlapping windows (2 s by default).

    Each window keeps a note of which recording it came from. That matters:
    windows from the same recording look alike, so if some land in training
    and others in testing, the test is too easy (data leakage).

    Returns
    -------
    windows : shape (n_windows, samples_per_window)
    window_sets : folder letter for each window
    window_groups : recording id for each window (used to keep recordings together)
    """
    n = int(round(window_sec * fs))
    per_recording = signals.shape[1] // n
    windows = signals[:, : per_recording * n].reshape(len(signals), per_recording, n)
    windows = windows.reshape(-1, n)
    window_sets = np.repeat(set_labels, per_recording)
    window_groups = np.repeat(recording_ids, per_recording)
    return windows, window_sets, window_groups

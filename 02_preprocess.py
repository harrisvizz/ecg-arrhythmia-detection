
import wfdb
import numpy as np
from scipy.signal import butter, sosfiltfilt
from pathlib import Path

# Project settings
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

RECORDS = [
    "100", "101", "102", "103", "104",
    "105", "106", "107", "108", "109",
    "111", "112", "113", "114", "115",
    "116", "117", "118", "119", "121",
    "122", "123", "124", "200", "201",
    "202", "203", "205", "207", "208",
    "209", "210", "212", "213", "214",
    "215", "217", "219", "220", "221",
    "222", "223", "228", "230", "231",
    "232", "233", "234"
]
BEFORE = 90
AFTER = 162

# Map MIT-BIH annotation symbols into five heartbeat classes

# Map MIT-BIH annotation symbols into five AAMI-style classes
CLASS_MAP = {
    # N: Normal and bundle branch block beats
    "N": "N",
    "L": "N",
    "R": "N",
    "e": "N",
    "j": "N",

    # S: Supraventricular ectopic beats
    "A": "S",
    "a": "S",
    "J": "S",
    "S": "S",

    # V: Ventricular ectopic beats
    "V": "V",
    "E": "V",

    # F: Fusion of ventricular and normal beats,
    # and fusion of paced and normal beats
    "F": "F",
    "f": "F",

    # Q: Paced beats and unknown beats
    "/": "Q",
    "Q": "Q",
}


def filter_ecg(signal, fs):
    """Reduce baseline drift and high-frequency noise."""
    sos = butter(
        4, [0.5, 40],
        btype="bandpass",
        fs=fs,
        output="sos"
    )
    return sosfiltfilt(sos, signal)

all_beats = []
all_labels = []
all_records = []

for record_id in RECORDS:
    print(f"Processing record {record_id}...")

    record = wfdb.rdrecord(record_id, pn_dir="mitdb", physical=True)
    annotations = wfdb.rdann(record_id, "atr", pn_dir="mitdb")

    signal = record.p_signal[:, 0]
    fs = record.fs

    # Filter the ECG signal
    filtered_signal = filter_ecg(signal, fs)

    # Extract a fixed-size segment around each annotated heartbeat
    for sample, symbol in zip(annotations.sample, annotations.symbol):
        if symbol not in CLASS_MAP:
            continue

        start = sample - BEFORE
        end = sample + AFTER

        # Skip beats too close to the beginning or end
        if start < 0 or end > len(filtered_signal):
            continue

        beat = filtered_signal[start:end]

        # Normalize each beat independently
        mean = np.mean(beat)
        std = np.std(beat)
        beat = (beat - mean) / (std + 1e-8)

        all_beats.append(beat.astype(np.float32))
        all_labels.append(CLASS_MAP[symbol])
        all_records.append(record_id)

# Convert results into NumPy arrays
X = np.array(all_beats, dtype=np.float32)
y = np.array(all_labels)
record_ids = np.array(all_records)

# Save the prepared dataset
np.savez_compressed(
    DATA_DIR / "ecg_beats.npz",
    X=X,
    y=y,
    record_ids=record_ids
)

print("\n--- Preprocessing Complete ---")
print("Heartbeat samples:", len(X))
print("Samples per heartbeat:", X.shape[1] if len(X) else 0)
print("Classes:", np.unique(y, return_counts=True))
print("Saved to:", DATA_DIR / "ecg_beats.npz")

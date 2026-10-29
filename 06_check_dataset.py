
import numpy as np
from pathlib import Path

data_path = Path("data/ecg_beats.npz")

if not data_path.exists():
    print("Dataset not found. Run 02_preprocess.py first.")
    raise SystemExit

data = np.load(data_path)
y = data["y"]
record_ids = data["record_ids"]

print("Total heartbeat samples:", len(y))
print("\nRecords included:")

for record in np.unique(record_ids):
    mask = record_ids == record
    print(f"Record {record}: {mask.sum()} heartbeat samples")

print("\nHeartbeat class distribution:")
classes, counts = np.unique(y, return_counts=True)

for label, count in zip(classes, counts):
    print(f"{label}: {count}")

print("\nDataset check complete.")


import wfdb
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Create the data folder if it does not exist
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Load ECG record 100 from the MIT-BIH Arrhythmia Database
print("Loading ECG data...")

record = wfdb.rdrecord("100", pn_dir="mitdb", physical=True)
annotation = wfdb.rdann("100", "atr", pn_dir="mitdb")

# Extract the first ECG channel
signal = record.p_signal[:, 0]
fs = record.fs

# Display basic information
print("\n--- ECG Record Information ---")
print("Record ID:", record.record_name)
print("Sampling frequency:", fs, "Hz")
print("Number of channels:", record.n_sig)
print("Signal duration:", round(len(signal) / fs / 60, 2), "minutes")
print("Number of annotations:", len(annotation.sample))
print("First 10 heartbeat labels:", annotation.symbol[:10])

# Plot the first five seconds of the ECG signal
samples = int(5 * fs)
time = np.arange(samples) / fs

plt.figure(figsize=(14, 4))
plt.plot(time, signal[:samples])
plt.xlabel("Time (seconds)")
plt.ylabel("ECG amplitude (mV)")
plt.title("MIT-BIH ECG Recording - Record 100")
plt.grid(True)
plt.tight_layout()
plt.show()


from pathlib import Path
import numpy as np
import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score
)

DATA_PATH = Path("data/ecg_beats.npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

# Load the prepared heartbeat data
if not DATA_PATH.exists():
    raise FileNotFoundError(
        "data/ecg_beats.npz not found. Run 02_preprocess.py first."
    )

data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]
record_ids = data["record_ids"]

print("Dataset shape:", X.shape)
print("Heartbeat classes:", np.unique(y, return_counts=True))

# IMPORTANT:
# This is a preliminary record-separated baseline.
# We hold out complete records rather than randomly splitting beats.
unique_records = np.unique(record_ids)

if len(unique_records) < 2:
    raise ValueError("At least two records are needed for this split.")

train_records, test_records = train_test_split(
    unique_records,
    test_size=0.2,
    random_state=42
)

train_mask = np.isin(record_ids, train_records)
test_mask = np.isin(record_ids, test_records)

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

print("Training records:", train_records)
print("Testing records:", test_records)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))

# Train a Random Forest
model = RandomForestClassifier(
    n_estimators=200,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
model.fit(X_train, y_train)

# Evaluate on held-out records
y_pred = model.predict(X_test)

print("\n--- Baseline Results ---")
print("Accuracy:", round(accuracy_score(y_test, y_pred), 4))
print("Macro F1:", round(f1_score(
    y_test, y_pred, average="macro", zero_division=0
), 4))

print("\nClassification Report:")
print(classification_report(
    y_test, y_pred, zero_division=0
))

# Save the model
model_path = MODEL_DIR / "random_forest_baseline.joblib"
joblib.dump(model, model_path)
print("Model saved to:", model_path)

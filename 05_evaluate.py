
from pathlib import Path
import numpy as np
import joblib
import torch
from torch import nn
import matplotlib.pyplot as plt
Path("outputs").mkdir(exist_ok=True)
from sklearn.metrics import (
    classification_report,
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score
)

# Load prepared data
data = np.load("data/ecg_beats.npz")
X, y, record_ids = data["X"], data["y"], data["record_ids"]

# Use the same record split as the training scripts
from sklearn.model_selection import train_test_split


records = np.unique(record_ids)

train_val_records, test_records = train_test_split(
    records,
    test_size=0.2,
    random_state=42
)

train_records, val_records = train_test_split(
    train_val_records,
    test_size=0.2,
    random_state=42
)

test_mask = np.isin(record_ids, test_records)

X_test, y_test = X[test_mask], y[test_mask]


# Evaluate Random Forest
rf_path = Path("models/random_forest_baseline.joblib")
if rf_path.exists():
    rf_model = joblib.load(rf_path)
    rf_pred = rf_model.predict(X_test)

    print("\n--- Random Forest ---")
    print("Accuracy:", accuracy_score(y_test, rf_pred))
    print("Macro F1:", f1_score(
        y_test, rf_pred, average="macro", zero_division=0
    ))
    print(classification_report(y_test, rf_pred, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y_test, rf_pred, xticks_rotation="vertical"
    )
    plt.title("Random Forest Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/random_forest_confusion_matrix.png")
    plt.show()
else:
    print("Random Forest model not found. Run 03_train_baseline.py first.")

# CNN architecture must match 04_train_cnn.py
class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, 7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

cnn_path = Path("models/ecg_cnn.pth")
if cnn_path.exists():
    checkpoint = torch.load(cnn_path, map_location="cpu")
    classes = checkpoint["classes"]
    cnn = ECGCNN(num_classes=len(classes))
    cnn.load_state_dict(checkpoint["model_state_dict"])
    cnn.eval()

    # Only use classes supported by the CNN's label mapping
    keep = np.isin(y_test, classes)
    X_cnn = torch.tensor(X_test[keep, None, :], dtype=torch.float32)
    y_cnn = y_test[keep]

    with torch.no_grad():
        pred_ids = cnn(X_cnn).argmax(1).numpy()

    cnn_pred = np.array(classes)[pred_ids]

    print("\n--- CNN ---")
    print("Accuracy:", accuracy_score(y_cnn, cnn_pred))
    print("Macro F1:", f1_score(
        y_cnn, cnn_pred, average="macro", zero_division=0
    ))
    print(classification_report(
        y_cnn, cnn_pred, labels=classes, zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_cnn, cnn_pred, labels=classes, xticks_rotation="vertical"
    )
    plt.title("CNN Confusion Matrix")
    plt.tight_layout()
    plt.savefig("outputs/cnn_confusion_matrix.png")
    plt.show()
else:
    print("CNN model not found. Run 04_train_cnn.py first.")

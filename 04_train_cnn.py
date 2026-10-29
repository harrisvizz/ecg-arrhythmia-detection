
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, f1_score

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = Path("data/ecg_beats.npz")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError("Run 02_preprocess.py first.")

data = np.load(DATA_PATH)
X = data["X"]
y = data["y"]
record_ids = data["record_ids"]

# Use the same record-based split as the baseline
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

print("Training records:", train_records)
print("Validation records:", val_records)
print("Testing records:", test_records)

train_mask = np.isin(record_ids, train_records)
val_mask = np.isin(record_ids, val_records)
test_mask = np.isin(record_ids, test_records)

classes = np.array(["F", "N", "Q", "S", "V"])

X_train, X_test = X[train_mask], X[test_mask]
y_train, y_test = y[train_mask], y[test_mask]

X_val, y_val = X[val_mask], y[val_mask]

print("\nTraining class distribution:")
print({c: int(np.sum(y_train == c)) for c in classes})

print("\nTesting class distribution:")
print({c: int(np.sum(y_test == c)) for c in classes})

# Convert class names into numeric labels
classes = np.array(["F", "N", "Q", "S", "V"])
class_to_id = {name: i for i, name in enumerate(classes)}

# Keep only classes present in the training set
train_keep = np.isin(y_train, classes)
val_keep = np.isin(y_val, classes)
test_keep = np.isin(y_test, classes)

X_train, y_train = X_train[train_keep], y_train[train_keep]
X_val, y_val = X_val[val_keep], y_val[val_keep]
X_test, y_test = X_test[test_keep], y_test[test_keep]

y_train_id = np.array([class_to_id[v] for v in y_train])
y_val_id = np.array([class_to_id[v] for v in y_val])
y_test_id = np.array([class_to_id[v] for v in y_test])

# CNN input shape: (samples, channels, signal length)
X_train_t = torch.tensor(X_train[:, None, :], dtype=torch.float32)
X_val_t = torch.tensor(X_val[:, None, :], dtype=torch.float32)
X_test_t = torch.tensor(X_test[:, None, :], dtype=torch.float32)

y_train_t = torch.tensor(y_train_id, dtype=torch.long)
y_val_t = torch.tensor(y_val_id, dtype=torch.long)
y_test_t = torch.tensor(y_test_id, dtype=torch.long)

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=128,
    shuffle=True
) 

class ECGCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, padding=3),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
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
        x = self.features(x)
        return self.classifier(x)

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=2.0):
        super().__init__()
        self.weight = weight
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.weight,
            reduction="none"
        )
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ECGCNN(num_classes=len(classes)).to(device)

# Give rarer classes more weight during training
# Calculate moderate class weights
class_to_id = {name: i for i, name in enumerate(classes)}
y_train_id = np.array([class_to_id[c] for c in y_train])

counts = np.bincount(y_train_id, minlength=len(classes))


# Moderately stronger class weighting experiment
weights = (
    len(y_train_id) / (len(classes) * np.maximum(counts, 1))
) ** 0.65

# Normalize weights so their average is 1
weights = weights / weights.mean()


print("\nClass weights:")
for name, weight in zip(classes, weights):
    print(f"{name}: {weight:.3f}")

weights_t = torch.tensor(
    weights,
    dtype=torch.float32,
    device=device
)

criterion = nn.CrossEntropyLoss(weight=weights_t)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("Device:", device)
print("Training samples:", len(X_train))
print("Testing samples:", len(X_test))
print("Training classes:", dict(zip(classes, counts)))

epochs = 20
best_val_f1 = -1.0
model_path = MODEL_DIR / "ecg_cnn.pth"

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * len(xb)

    # Evaluate on validation records
    model.eval()
    with torch.no_grad():
        val_logits = model(X_val_t.to(device))
        val_predictions = val_logits.argmax(dim=1).cpu().numpy()

    val_f1 = f1_score(
        y_val_id,
        val_predictions,
        labels=np.arange(len(classes)),
        average="macro",
        zero_division=0
    )

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"- Loss: {running_loss / len(X_train):.4f} "
        f"- Validation Macro F1: {val_f1:.4f}"
    )

    # Save the checkpoint with the best validation macro F1
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "classes": classes.tolist(),
                "best_val_f1": best_val_f1
            },
            model_path
        )
        print("Best model checkpoint saved.")

print(f"\nBest validation Macro F1: {best_val_f1:.4f}")
print("Best CNN model saved to:", model_path)

# Load the best checkpoint before final test evaluation
checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Evaluate on the held-out records
with torch.no_grad():
    logits = model(X_test_t.to(device))
    predictions = logits.argmax(dim=1).cpu().numpy()


print("\n--- CNN Results ---")
print("Accuracy:", round(accuracy_score(y_test_id, predictions), 4))
print("Macro F1:", round(f1_score(
    y_test_id, predictions, average="macro", zero_division=0
), 4))
print("\nClassification Report:")
print(classification_report(
    y_test_id,
    predictions,
    labels=np.arange(len(classes)),
    target_names=classes,
    zero_division=0
))


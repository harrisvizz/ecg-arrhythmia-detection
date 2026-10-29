
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


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
        )

        self.classifier = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


def main():
    model_path = Path("models/ecg_cnn.pth")
    data_path = Path("data/ecg_beats.npz")

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    # Load the trained CNN
    checkpoint = torch.load(model_path, map_location="cpu")
    classes = checkpoint["classes"]

    model = ECGCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Load the prepared ECG heartbeat dataset
    with np.load(data_path) as data:
        X = data["X"]
        y = data["y"]

        print(f"\nTotal heartbeat samples: {len(X)}")
        print("Available classes: F, N, Q, S, V")

        try:
            heartbeat_index = int(
                input("Enter heartbeat index (0 to {}): ".format(len(X) - 1))
            )
        except ValueError:
            print("Invalid input. Please enter a whole number.")
            return

        if heartbeat_index < 0 or heartbeat_index >= len(X):
            print(f"Index must be between 0 and {len(X) - 1}.")
            return

        heartbeat = X[heartbeat_index]
        actual_class = y[heartbeat_index]

    # Convert to shape (batch, channels, signal length)
    input_tensor = torch.tensor(
        heartbeat, dtype=torch.float32
    ).unsqueeze(0).unsqueeze(0)

    # Predict the heartbeat class
    with torch.no_grad():
        logits = model(input_tensor)
        probabilities = torch.softmax(logits, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()

    predicted_class = classes[predicted_index]
    score = probabilities[0, predicted_index].item()

    print("\n--- ECG Heartbeat Prediction ---")
    print(f"Heartbeat index : {heartbeat_index}")
    print(f"Actual class    : {actual_class}")
    print(f"Predicted class : {predicted_class}")
    print(f"Softmax score   : {score:.2%}")
    print(
        "Prediction match:",
        "Yes" if str(predicted_class) == str(actual_class) else "No"
    )
    print("\nNote: This is an educational model, not a clinical diagnostic tool.")


if __name__ == "__main__":
    main()


from pathlib import Path

import streamlit as st
import numpy as np
import joblib
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# --------------------------------------------------
# 1. PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="ECG Arrhythmia Detection",
    page_icon="❤️",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "ecg_beats.npz"
RF_PATH = BASE_DIR / "models" / "random_forest_baseline.joblib"
CNN_PATH = BASE_DIR / "models" / "ecg_cnn.pth"
OUTPUT_DIR = BASE_DIR / "outputs"

CLASS_NAMES = {
    "F": "Fusion Beat",
    "N": "Normal / Grouped Beat",
    "Q": "Unknown / Paced Beat",
    "S": "Supraventricular Ectopic Beat",
    "V": "Ventricular Ectopic Beat"
}

# --------------------------------------------------
# 2. CNN ARCHITECTURE
# Must match the training architecture
# --------------------------------------------------

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
        return self.classifier(self.features(x))


# --------------------------------------------------
# 3. LOAD DATA AND MODELS
# --------------------------------------------------

@st.cache_data
def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            "data/ecg_beats.npz is missing."
        )

    with np.load(DATA_PATH) as data:
        X = data["X"].astype(np.float32)
        y = data["y"].astype(str)

    return X, y


@st.cache_resource
def load_random_forest():
    if not RF_PATH.exists():
        raise FileNotFoundError(
            "Random Forest model not found."
        )
    return joblib.load(RF_PATH)


@st.cache_resource
def load_cnn():
    if not CNN_PATH.exists():
        raise FileNotFoundError("CNN checkpoint not found.")

    checkpoint = torch.load(
        CNN_PATH,
        map_location="cpu",
        weights_only=True
    )

    classes = checkpoint["classes"]

    model = ECGCNN(num_classes=len(classes))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model, classes


# --------------------------------------------------
# 4. SIDEBAR NAVIGATION
# --------------------------------------------------

st.sidebar.title("❤️ ECG Analytics")
page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard Overview",
        "ECG Explorer",
        "Model Prediction",
        "Model Comparison",
        "About Project"
    ]
)

# --------------------------------------------------
# 5. LOAD PROJECT RESOURCES
# --------------------------------------------------

try:
    X, y = load_dataset()
    rf_model = load_random_forest()
    cnn_model, cnn_classes = load_cnn()

except Exception as error:
    st.error(f"Unable to load project resources: {error}")
    st.info(
        "Check that the data and both trained models "
        "are present in their respective folders."
    )
    st.stop()


# --------------------------------------------------
# 6. DASHBOARD OVERVIEW
# --------------------------------------------------

if page == "Dashboard Overview":

    st.title("ECG Heart Arrhythmia Detection")
    st.write(
        "An interactive heartbeat classification system "
        "using Machine Learning and Deep Learning."
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Heartbeats", f"{len(X):,}")
    col2.metric("Signal Length", X.shape[1])
    col3.metric("Heartbeat Classes", len(np.unique(y)))
    col4.metric("Models Available", "2")

    st.subheader("Project Information")

    left, right = st.columns(2)

    with left:
        st.markdown("### Dataset")
        st.write("MIT-BIH Arrhythmia Database")
        st.write("Sampling frequency: 360 Hz")
        st.write("Preprocessed heartbeat segments: 252 samples each")

    with right:
        st.markdown("### Models")
        st.write("Random Forest Classifier")
        st.write("1D Convolutional Neural Network")

    st.subheader("Heartbeat Class Distribution")

    labels, counts = np.unique(y, return_counts=True)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, counts)
    ax.set_xlabel("Heartbeat Class")
    ax.set_ylabel("Number of Samples")
    ax.set_title("Dataset Class Distribution")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# 7. ECG EXPLORER
# --------------------------------------------------

elif page == "ECG Explorer":

    st.title("ECG Signal Explorer")
    st.write("Explore individual preprocessed ECG heartbeats.")

    index = st.number_input(
        "Select heartbeat index",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
        step=1
    )

    heartbeat = X[int(index)]
    actual_class = y[int(index)]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(heartbeat)
    ax.set_title(f"ECG Heartbeat - Sample {index}")
    ax.set_xlabel("Time Samples")
    ax.set_ylabel("Normalized Amplitude")
    ax.grid(True)

    st.pyplot(fig)
    plt.close(fig)

    col1, col2 = st.columns(2)

    col1.metric("Heartbeat Index", int(index))
    col2.metric("Actual Class", actual_class)

    st.info(CLASS_NAMES.get(actual_class, "Other"))


# --------------------------------------------------
# 8. MODEL PREDICTION
# --------------------------------------------------

elif page == "Model Prediction":

    st.title("Heartbeat Classification")
    st.write("Select a heartbeat and choose a trained model.")

    index = st.number_input(
        "Heartbeat index",
        min_value=0,
        max_value=len(X) - 1,
        value=0,
        step=1
    )

    model_choice = st.selectbox(
        "Select prediction model",
        ["Random Forest", "1D CNN"]
    )

    heartbeat = X[int(index)]
    actual_class = y[int(index)]

    if st.button("Predict Heartbeat", type="primary"):

        if model_choice == "Random Forest":

            probabilities = rf_model.predict_proba(
                heartbeat.reshape(1, -1)
            )[0]

            class_order = rf_model.classes_
            predicted_class = class_order[
                np.argmax(probabilities)
            ]

        else:

            input_tensor = torch.tensor(
                heartbeat,
                dtype=torch.float32
            ).unsqueeze(0).unsqueeze(0)

            with torch.no_grad():
                logits = cnn_model(input_tensor)
                probabilities = torch.softmax(
                    logits, dim=1
                ).numpy()[0]

            predicted_class = cnn_classes[
                int(np.argmax(probabilities))
            ]

            class_order = cnn_classes

        score = float(np.max(probabilities))

        st.divider()
        st.subheader("Prediction Result")

        col1, col2, col3 = st.columns(3)

        col1.metric("Actual Class", actual_class)
        col2.metric("Predicted Class", predicted_class)
        col3.metric("Model Output Score", f"{score:.2%}")

        if predicted_class == actual_class:
            st.success("Prediction matches the dataset label.")
        else:
            st.warning("Prediction does not match the dataset label.")

        st.write(
            "**Predicted category:**",
            CLASS_NAMES.get(predicted_class, "Unknown")
        )

        st.subheader("Class-wise Model Scores")

        fig, ax = plt.subplots(figsize=(9, 4))
        ax.bar(
            [str(c) for c in class_order],
            probabilities
        )
        ax.set_xlabel("Heartbeat Class")
        ax.set_ylabel("Model Output Score")
        ax.set_ylim(0, 1)
        ax.set_title("Prediction Scores")

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "These are model output scores, not clinically "
            "validated probabilities or diagnostic confidence."
        )

        st.subheader("Selected ECG Signal")

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(heartbeat)
        ax.set_xlabel("Time Samples")
        ax.set_ylabel("Normalized Amplitude")
        ax.grid(True)

        st.pyplot(fig)
        plt.close(fig)


# --------------------------------------------------
# 9. MODEL COMPARISON
# --------------------------------------------------

elif page == "Model Comparison":

    st.title("Model Performance Comparison")

    st.write(
        "Performance values from the project's held-out "
        "record-based evaluation."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Random Forest")
        st.metric("Accuracy", "92.07%")
        st.metric("Macro F1-Score", "0.5121")

    with col2:
        st.subheader("1D CNN")
        st.metric("Accuracy", "85.53%")
        st.metric("Macro F1-Score", "0.4584")

    st.subheader("Class-wise F1-Score")

    classes = ["F", "N", "Q", "S", "V"]

    rf_scores = [0.01, 0.96, 0.95, 0.00, 0.64]
    cnn_scores = [0.03, 0.93, 0.69, 0.08, 0.57]

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(classes))
    width = 0.35

    ax.bar(x - width / 2, rf_scores, width, label="Random Forest")
    ax.bar(x + width / 2, cnn_scores, width, label="1D CNN")

    ax.set_xticks(x)
    ax.set_xticklabels(classes)
    ax.set_ylabel("F1-Score")
    ax.set_ylim(0, 1)
    ax.set_title("Class-wise Model Performance")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)

    st.subheader("Confusion Matrices")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Random Forest")
        image_path = OUTPUT_DIR / "random_forest_confusion_matrix.png"
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)

    with col2:
        st.write("1D CNN")
        image_path = OUTPUT_DIR / "cnn_confusion_matrix.png"
        if image_path.exists():
            st.image(str(image_path), use_container_width=True)

    st.warning(
        "Both models struggle with minority classes, particularly "
        "F and S. Accuracy should not be interpreted in isolation."
    )


# --------------------------------------------------
# 10. ABOUT THE PROJECT
# --------------------------------------------------

elif page == "About Project":

    st.title("About the Project")

    st.markdown("""
    ### Heart Arrhythmia Detection Using ECG Signals

    **Dataset:** MIT-BIH Arrhythmia Database

    **Signal preprocessing:**
    - Bandpass filtering: 0.5–40 Hz
    - Heartbeat segmentation: 252 samples
    - Individual heartbeat normalization

    **Machine Learning:**
    - Random Forest classifier

    **Deep Learning:**
    - One-dimensional Convolutional Neural Network

    **Evaluation metrics:**
    - Accuracy
    - Precision
    - Recall
    - Macro F1-score
    - Confusion matrix

    **Heartbeat classes:**
    - F: Fusion beat
    - N: Normal and grouped beats
    - Q: Unknown or paced beats
    - S: Supraventricular ectopic beat
    - V: Ventricular ectopic beat
    """)

    st.info(
        "This application is an educational machine learning "
        "demonstration. It is not a medical diagnostic system."
    )


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "ECG Arrhythmia Detection | Machine Learning + Deep Learning "
    "| Educational Prototype"
)

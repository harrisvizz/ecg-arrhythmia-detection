
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Heart Arrhythmia Detection",
    page_icon="🤖",
    layout="wide"
)

# Main title
st.title("Heart Arrhythmia Detection Using ECG Signals")

st.markdown("---")

# Introduction
st.header("Welcome to Heart Arrhythmia Detection")

st.write(
    """
    This application is developed to analyze Electrocardiogram (ECG)
    signals and identify potential heart arrhythmia patterns using
    Machine Learning and Deep Learning techniques.
    """
)

# About the project
st.subheader("📌 About the Project")

st.write(
    """
    Heart arrhythmia refers to an irregular heartbeat that occurs
    when the electrical signals controlling the heartbeat do not
    function properly.

    This project uses ECG signals to detect and classify different
    types of heart rhythm abnormalities using trained machine
    learning models.
    """
)

# Project objectives
st.subheader("🎯 Project Objectives")

st.markdown(
    """
    - Analyze ECG signals.
    - Preprocess ECG data for model training.
    - Extract meaningful patterns from ECG signals.
    - Apply Machine Learning and CNN techniques.
    - Classify ECG signals based on learned patterns.
    """
)

# Project workflow
st.subheader("⚙️ Project Workflow")

st.markdown(
    """
    1. ECG Signal Data Collection
    2. Data Exploration
    3. ECG Signal Preprocessing
    4. Model Training
    5. Arrhythmia Classification
    """
)

st.markdown("---")

# Footer
st.info(
    "This project is developed for academic and research purposes. "
    "It is not intended to replace professional medical diagnosis."
)

st.success("Heart Arrhythmia Detection application is running successfully!")

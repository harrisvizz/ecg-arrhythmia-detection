# Heart Arrhythmia Detection Using ECG Signals

## Project Overview

This project focuses on detecting and classifying different types of heartbeats using Electrocardiogram (ECG) signals and machine learning techniques. It uses the **MIT-BIH Arrhythmia Database** to develop and evaluate two classification models: a Random Forest baseline model and a one-dimensional Convolutional Neural Network (1D CNN).

The project demonstrates the complete machine learning workflow, including ECG signal preprocessing, exploratory data analysis, model training, performance evaluation, confusion matrix visualization, and individual heartbeat prediction.

This is an educational project intended to explore the application of machine learning and deep learning to biomedical signal classification.

## Objectives

- Explore and understand ECG heartbeat data.
- Preprocess ECG signals to prepare them for model training.
- Develop a Random Forest model as a machine learning baseline.
- Build a 1D CNN to learn patterns from ECG heartbeat signals.
- Evaluate model performance using accuracy, macro F1-score, and per-class classification metrics.
- Visualize classification results using confusion matrices.
- Predict the class of an individual heartbeat using a trained CNN model.

## Dataset

**Dataset:** MIT-BIH Arrhythmia Database

The dataset contains ECG recordings used for studying and classifying cardiac arrhythmias.

The preprocessing pipeline extracts individual heartbeat segments and groups them into five classes:

| Class | Description |
|---|---|
| F | Fusion of ventricular and normal beat |
| N | Normal beat and other beats grouped into the N category |
| Q | Unknown, paced, or other beats grouped into the Q category |
| S | Supraventricular ectopic beat |
| V | Ventricular ectopic beat |

After preprocessing, the dataset contains:

- **Total heartbeat samples:** 109,452
- **Samples per heartbeat:** 252
- **Number of classes:** 5
- **Class distribution:** Imbalanced, with the N class representing the majority of samples.

The dataset is divided by ECG record rather than randomly splitting individual heartbeat samples. This helps reduce data leakage between training and testing sets.

**Note:** The class labels follow the grouping implemented in this project and should not be interpreted as a complete clinical classification system.

## Technologies Used

- **Programming Language:** Python
- **Development Environment:** Visual Studio Code
- **Data Processing:** NumPy, Pandas
- **Signal Processing:** SciPy, WFDB
- **Machine Learning:** Scikit-learn
- **Deep Learning:** PyTorch
- **Visualization:** Matplotlib
- **Model Storage:** Joblib and PyTorch model checkpoints

## Project Structure

```text
ecg-arrhythmia-detection/
│
├── data/
│   └── ecg_beats.npz
│
├── models/
│   ├── random_forest_baseline.joblib
│   └── ecg_cnn.pth
│
├── outputs/
│   ├── random_forest_confusion_matrix.png
│   └── cnn_confusion_matrix.png
│
├── 01_explore_data.py
├── 02_preprocess.py
├── 03_train_baseline.py
├── 04_train_cnn.py
├── 05_evaluate.py
├── 06_predict.py
│
├── requirements.txt
└── README.md
```

## Project Workflow

### 1. Data Exploration

The `01_explore_data.py` script explores the dataset and helps understand the ECG heartbeat samples, their labels, and class distribution.

### 2. Data Preprocessing

The `02_preprocess.py` script prepares ECG signals for model training.

The preprocessing steps include:

- Loading ECG recordings and heartbeat annotations.
- Extracting heartbeat segments.
- Applying a 0.5–40 Hz bandpass filter.
- Normalizing individual heartbeat segments.
- Grouping heartbeat annotations into five classes.
- Saving the processed dataset as `data/ecg_beats.npz`.

### 3. Random Forest Baseline Model

The `03_train_baseline.py` script trains a Random Forest classifier and compares its performance with other traditional machine learning algorithms.

The baseline experiments include Logistic Regression, Decision Tree, Random Forest, and Gradient Boosting.

The trained Random Forest model is saved as:

`models/random_forest_baseline.joblib`

### 4. CNN Model Training

The `04_train_cnn.py` script trains a one-dimensional Convolutional Neural Network using the preprocessed ECG heartbeat signals.

The CNN uses convolutional layers to learn patterns from the signals, followed by a classifier to predict one of the five heartbeat classes.

The training process includes:

- Splitting data by ECG record into training, validation, and testing sets.
- Applying class-weighted loss to help address class imbalance.
- Monitoring validation macro F1-score.
- Saving the best model checkpoint based on validation macro F1-score.

The trained model is saved as:

`models/ecg_cnn.pth`

### 5. Model Evaluation

The `05_evaluate.py` script evaluates the trained Random Forest and CNN models on the held-out test records.

It reports classification metrics and generates confusion matrices for both models.

Generated visualizations:

- `outputs/random_forest_confusion_matrix.png`
- `outputs/cnn_confusion_matrix.png`

### 6. Individual Heartbeat Prediction

The `06_predict.py` script loads the trained CNN model and allows the user to select a heartbeat by its index.

The script displays:

- Selected heartbeat index
- Actual class
- Predicted class
- Model softmax score
- Whether the prediction matches the dataset label

The softmax score represents the model's output for the predicted class. It is not a medically validated confidence score.

## Model Evaluation and Results

The models were evaluated using a record-based train/test split. The following results were obtained from the current experiments.

### Overall Performance

| Metric | Random Forest | 1D CNN |
|---|---:|---:|
| Accuracy | 92.07% | 85.53% |
| Macro F1-score | 0.5121 | 0.4584 |

### Per-Class F1-Scores

| Heartbeat Class | Random Forest | 1D CNN |
|---|---:|---:|
| F | 0.01 | 0.03 |
| N | 0.96 | 0.93 |
| Q | 0.95 | 0.69 |
| S | 0.00 | 0.08 |
| V | 0.64 | 0.57 |

### Results Interpretation

- The Random Forest model achieved higher overall accuracy than the CNN in the current experiments.
- Both models performed well on the majority N class.
- Both models struggled to classify the minority F and S classes.
- The CNN achieved a slightly higher F1-score for the F and S classes, but its performance on these classes remained poor.
- The difference between accuracy and macro F1-score highlights the effect of class imbalance. Accuracy alone does not adequately describe performance across all heartbeat classes.

These results indicate that further experimentation is required to improve minority-class detection and generalization across different ECG records.

## Key Observations

1. ECG signal preprocessing is an important step in preparing heartbeat data for classification.
2. Class imbalance significantly affects model performance.
3. Record-based splitting provides a more meaningful evaluation than randomly splitting individual beats from the same recordings.
4. A high overall accuracy does not necessarily indicate good performance across every class.
5. The softmax score produced during prediction should not be treated as proof that a prediction is correct.
6. The Random Forest model provides a useful baseline for comparison with the deep learning approach.

## How to Run the Project

### Prerequisites

- Python installed on your system.
- Visual Studio Code or another Python development environment.
- An internet connection for downloading the dataset if it is not already available.

### Step 1: Clone or Download the Project

Download the repository or open the project folder in Visual Studio Code.

### Step 2: Create and Activate a Virtual Environment

Create a virtual environment:

```bash
python -m venv .venv
```

On Windows Command Prompt:

```bash
.venv\Scripts\activate
```

On Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies

Install the required packages:

```bash
pip install -r requirements.txt
```

### Step 4: Explore the Dataset

```bash
python 01_explore_data.py
```

### Step 5: Preprocess the ECG Signals

```bash
python 02_preprocess.py
```

This step creates the processed dataset at `data/ecg_beats.npz`.

### Step 6: Train the Random Forest Model

```bash
python 03_train_baseline.py
```

### Step 7: Train the CNN Model

```bash
python 04_train_cnn.py
```

This step trains the CNN and saves the best checkpoint to `models/ecg_cnn.pth`.

### Step 8: Evaluate the Models

```bash
python 05_evaluate.py
```

The evaluation script reports performance metrics and saves the confusion matrix images in the `outputs/` directory.

### Step 9: Predict an Individual Heartbeat

```bash
python 06_predict.py
```

Enter a heartbeat index when prompted. The valid range is `0` to `109451` for the current processed dataset.

The script displays the actual class, predicted class, softmax score, and whether the prediction matches the dataset label.

**Important:** Run the scripts in the order shown when building the project from scratch. The prediction and evaluation scripts require the processed dataset and trained model files to exist.

## Limitations

- The dataset has an imbalanced class distribution, which affects minority-class classification.
- The current models have poor F1-scores for some heartbeat classes, particularly F and S.
- Performance on this dataset does not establish performance on ECG recordings from other populations, devices, or clinical settings.
- The softmax score is not a calibrated or clinically validated measure of certainty.
- The models have not been validated for clinical use.
- Further work is needed to improve class balance, feature representation, model generalization, and evaluation across independent datasets.

## Future Improvements

- Experiment with alternative class-balancing methods and loss functions.
- Explore signal augmentation techniques for underrepresented classes.
- Improve the CNN architecture and training strategy.
- Compare additional machine learning and deep learning models.
- Perform hyperparameter tuning using validation data.
- Evaluate generalization on independent ECG datasets.
- Develop a visualization interface for inspecting heartbeat signals and predictions.
- Investigate model interpretability techniques to understand which signal patterns influence predictions.

## Disclaimer

This project is intended for educational and research purposes only. It is not a clinically validated diagnostic system and must not be used for medical diagnosis, treatment decisions, or other clinical decision-making.

## Acknowledgements

The project uses the MIT-BIH Arrhythmia Database for educational experimentation with ECG signal processing and heartbeat classification.

The dataset is associated with PhysioNet. Please refer to the original dataset documentation and terms of use when using or redistributing the data.

## Author

**Shree Harris S D**

# ⚡ Transmission Line Fault Detection Using Machine Learning

**Course:** AI4003 – Applied Machine Learning  
**Assignment:** Assignment 2  
**Project:** Detecting Faults on a Three-Phase Transmission Line

## 📌 Project Overview

This project applies machine learning to classify electrical transmission-line measurements as **fault** or **no fault**. The objective is to explore current and voltage measurements, implement Logistic Regression, evaluate Support Vector Machine (SVM) models, and build a Streamlit web interface for interactive predictions.

The project compares models using test accuracy, confusion-matrix values, false alarms, and missed faults.

> **Note:** This is an academic demonstration, not a certified electrical protection system.

## 🎯 Objectives

- Explore electrical fault-detection data using visualizations.
- Implement Logistic Regression manually using NumPy.
- Compare raw standardized features (Set R) with engineered features (Set Q).
- Train Linear and RBF Support Vector Machine models.
- Evaluate final models using a held-out test set.
- Build a Streamlit interface for entering measurements and comparing predictions.

## 📊 Dataset

The project uses an electrical fault-detection dataset based on simulated transmission-line measurements.

**Input features:**

- Phase currents: `Ia`, `Ib`, `Ic`
- Phase voltages: `Va`, `Vb`, `Vc`

**Target:**

- `0` — No fault
- `1` — Fault

The assignment dataset contains approximately 12,000 rows. The reported dataset summary includes 5,496 fault samples and 6,505 normal samples after removing empty columns. Verify these counts against your final processed dataset before publishing.

## 🧠 Models Implemented

### 1. Logistic Regression

A manual implementation using NumPy, including:

- Sigmoid activation function
- Binary cross-entropy loss
- Batch gradient descent
- Learning-rate experiments
- Comparison of Set R and Set Q features

### 2. Support Vector Machine (SVM)

Models implemented using `sklearn.svm.SVC`:

- Linear SVM
- RBF SVM
- Experiments with different values of `C`
- Experiments with different RBF `gamma` values
- Validation-based model selection

## 📈 Test Results

The following results were reported for the 2,401-row test set.

| Model | Accuracy | TN | FP | FN | TP |
|---|---:|---:|---:|---:|---:|
| Logistic Regression — Set R | 72.64% | 1,305 | 0 | 657 | 439 |
| Logistic Regression — Set Q | 97.88% | 1,299 | 6 | 45 | 1,051 |
| Linear SVM (`C=0.1`) | 73.59% | 1,305 | 0 | 634 | 462 |
| **RBF SVM (`C=100`, `gamma=1`)** | **99.58%** | **1,301** | **4** | **6** | **1,090** |

**Confusion-matrix terminology:**

- **TN:** Normal samples correctly classified as normal.
- **FP:** Normal samples incorrectly classified as faults (false alarms).
- **FN:** Fault samples incorrectly classified as normal (missed faults).
- **TP:** Fault samples correctly classified as faults.

The RBF SVM achieved the highest reported test accuracy and had six missed faults and four false alarms. However, a high random-split test score does not establish real-world protection-system reliability.

## 🔍 Key Findings

- Feature engineering substantially improved Logistic Regression performance in the reported experiments.
- The Linear SVM and raw-feature Logistic Regression produced similar results.
- The RBF SVM performed best among the four reported final models.
- False alarms and missed faults are important metrics in addition to accuracy.
- Because neighboring time samples can be highly correlated, a random train/test split may produce optimistic results. A time-based or event-based evaluation would provide a stronger assessment of generalization.

## 🖥️ Streamlit Web Application

The application allows users to enter the six phase-current and phase-voltage measurements and view predictions from the saved Set Q Logistic Regression and RBF SVM models.

### Application Screenshot

Add a screenshot of your working Streamlit application here.

For example, save the screenshot as `images/streamlit-app.png` and display it with:

```markdown
![Streamlit Application](images/streamlit-app.png)
```

## 🗂️ Project Structure

The repository may be organized as follows. Adjust the paths to match your actual files.

```text
fault-detection-project/
├── fault-detection/
│   ├── app/
│   │   └── app.py
│   ├── data/
│   │   └── detect_dataset.csv
│   ├── models/
│   ├── results/
│   │   └── analysis.md
│   └── notebooks/
├── images/
│   └── streamlit-app.png
├── requirements.txt
└── README.md
```

**Important:** Only include files and directories that actually exist in your repository. Do not upload large datasets or trained models unless permitted and appropriate.

## ⚙️ Installation and Setup

### Prerequisites

- Python 3.10 or a compatible Python version
- Visual Studio Code
- Git
- Required Python packages listed in `requirements.txt`

### 1. Clone the Repository

```bash
git clone <YOUR-GITHUB-REPOSITORY-URL>
cd fault-detection-project
```

Replace the placeholder with your actual GitHub repository URL.

### 2. Create a Virtual Environment

**Windows:**

```bash
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If you use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If `requirements.txt` is missing, create it using the packages required by your code. Typical dependencies for this project include:

```text
numpy
pandas
matplotlib
scikit-learn
streamlit
joblib
```

Add or remove packages according to the actual imports in your project.

## ▶️ Run the Streamlit Application

From the repository root, run:

```bash
streamlit run .\fault-detection\app\app.py
```

Alternatively, enter the project directory first:

```powershell
cd .\fault-detection
streamlit run .\app\app.py
```

Streamlit will display a local URL in the terminal, usually:

```text
http://localhost:8501
```

Open that URL in your browser to use the application.

**Before running:** Make sure the saved model files and dataset paths referenced by `app.py` exist and are correctly configured.

## 📚 Analysis and Documentation

The project analysis covers:

1. Comparison of Logistic Regression and Linear SVM.
2. Interpretation of squared-current and squared-voltage feature weights.
3. Comparison of support-vector counts.
4. The effect of the SVM regularization parameter `C`.
5. Limitations of a random split on consecutive time-series samples.
6. Model selection based on missed faults and false alarms.

See `fault-detection/results/analysis.md` if this file is included in your repository.

## 🚀 Future Improvements

- Evaluate on unseen fault events using a time-based split.
- Test performance across different fault types and operating conditions.
- Add precision, recall, F1-score, and ROC-AUC analysis.
- Improve model robustness and investigate class imbalance.
- Test latency and reliability before considering any real-world application.

## ⚠️ Limitations

The reported results are specific to the dataset and experimental split used in this assignment. They should not be interpreted as proof of safety or reliability on real transmission lines. Actual deployment would require extensive validation using representative fault events, hardware testing, and assessment by qualified power-system protection engineers.

## 👨‍💻 Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
- Git and GitHub

## 📄 Academic Disclaimer

This repository is for educational and academic purposes as part of an Applied Machine Learning assignment. It is not intended to replace certified protection relays or professional electrical engineering systems.

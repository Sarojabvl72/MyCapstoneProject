# MyCapstoneProject

<p align="center">📊 EDA & Fraud Detection Notebook — Comprehensive analysis of bank transaction data</p>

---

## Table of Contents

- [Overview](#overview)
- [Project structure](#project-structure)
- [Dataset](#dataset)
- [Fraud logic](#fraud-logic)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature engineering](#feature-engineering)
- [Machine learning models](#machine-learning-models)
- [Results](#results)
- [Requirements](#requirements)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Overview

This repository contains a Jupyter Notebook that performs Exploratory Data Analysis (EDA) and trains multiple machine-learning models to detect synthetic fraud flags on a structured bank transaction dataset. The notebook demonstrates data inspection, rule-based target creation, feature engineering, model training, evaluation, and visualization for interpretability.

Dataset size used in the notebook: 2,512 transactions (synthetic or anonymized for demonstration).

---

## Project structure

- `notebook.ipynb` — main analysis and modeling notebook (rename if different)
- `data/` — directory for datasets (CSV files)
- `requirements.txt` — Python dependencies
- `README.md` — this file

---

## Dataset

Expected columns (adjust names in the notebook if required):

- `transaction_id`, `customer_id`
- `transaction_type` (Credit / Debit)
- `timestamp`
- `amount`
- `location` (city / state)
- `ip_address`
- `device_id`
- `age`
- `login_attempts`
- `transaction_duration_seconds`

---

## Fraud logic

A synthetic `FraudFlag` (0 = normal, 1 = fraud) is created using strict, demonstrative business rules. A transaction is flagged as fraud when ALL of the following apply:

1. Transaction pattern: a **Credit** followed by an immediate **Debit** (time difference ≤ 300 seconds).
2. Location: not recognized as a U.S. city.
3. IP address: outside typical U.S. IP blocks.
4. Customer age: between **20 and 35** (inclusive).
5. Login attempts: **> 3** attempts.
6. Transaction duration: **< 30 seconds**.

These rules are intentionally narrow to generate a clear target for supervised learning and to illustrate model evaluation on a defined signal.

---

## Exploratory Data Analysis (EDA)

The notebook performs:

- Data loading and schema inspection
- Summary statistics and missing-value analysis
- Distribution plots (histograms, boxplots) for numeric features
- Correlation heatmap
- Time-based transaction behavior analysis

Visualizations use matplotlib and seaborn (optional) for clarity.

---

## Feature engineering

Key preprocessing steps in the notebook:

- Parse datetime columns and compute time differences between consecutive transactions
- Create the synthetic `FraudFlag` using the rule set above
- One-hot encode categorical variables (e.g., `transaction_type`)
- Drop non-modeling identifiers (e.g., `transaction_id`, `customer_id`) before training
- Train/test split using stratification on the fraud label

---

## Machine learning models

The notebook trains and compares multiple classifiers:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest
- Bagging Classifier
- Voting Classifier (soft voting)

Each model reports accuracy, precision/recall/F1 (classification report), and predicted class distribution. A sample decision tree is visualized for interpretability.

---

## Results

The notebook prints and visualizes evaluation metrics for each model. For fraud detection tasks, pay special attention to precision and recall for the fraud (positive) class because fraud is typically the minority class.

---

## Requirements

Install dependencies with pip:

```bash
pip install pandas numpy matplotlib scikit-learn seaborn
```

(Optionally create a `requirements.txt` for reproducibility.)

---

## Usage

1. Clone the repository and open the notebook:

```bash
git clone https://github.com/Sarojabvl72/MyCapstoneProject.git
cd MyCapstoneProject
jupyter lab   # or jupyter notebook
```

2. Install dependencies and run the notebook cells in order. Update dataset paths in the notebook if your files are stored elsewhere.

---

## Contributing

Contributions are welcome. Please open an issue or submit a pull request with clear descriptions and tests or reproducible examples when appropriate.

---

## License

Add a license (e.g., MIT) to clarify reuse and distribution terms.

---

## Contact

Repository owner: `Sarojabvl72`

For questions or feedback, open an issue in this repository.

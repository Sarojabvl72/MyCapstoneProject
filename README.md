# MyCapstoneProject
# 📊 EDA & Fraud Detection Notebook  
### Comprehensive Analysis of Bank Transaction Data

This repository contains a Jupyter Notebook designed to perform **Exploratory Data Analysis (EDA)** and build a **fraud detection model** using a structured dataset of bank transactions. The notebook walks through data exploration, feature engineering, fraud‑flag creation based on business rules, and multiple machine‑learning models for classification.

---

## 📁 Project Structure


---

## 📘 Overview

The notebook analyzes a dataset of **2,512 bank transactions**, including customer demographics, transaction metadata, device/IP information, and behavioral indicators such as login attempts and transaction duration.

The project includes:

- Data loading and inspection  
- Exploratory Data Analysis (EDA)  
- Fraud‑flag creation using custom business rules  
- Feature encoding  
- Model training and evaluation  
- Visualization of decision‑tree logic  

---

## 🧠 Fraud Logic Used

Fraud is flagged when **all** of the following conditions are met:

1. **Transaction Pattern**
   - A **Credit** transaction followed by an **immediate Debit**  
   - Time difference ≤ 300 seconds

2. **Suspicious Location**
   - Location **not recognized as a U.S. city**

3. **Suspicious IP Address**
   - IP address **not belonging to typical U.S. IP blocks**

4. **Customer Age**
   - Between **20 and 35**

5. **Login Attempts**
   - More than **3 attempts**

6. **Transaction Duration**
   - Less than **30 seconds**

These rules generate a synthetic `FraudFlag` column (0 = normal, 1 = fraud).

---

## 🔍 Exploratory Data Analysis (EDA)

The notebook performs:

- Dataset shape, schema, and summary statistics  
- Missing‑value analysis  
- Histograms and boxplots for numerical features  
- Correlation heatmap  
- Time‑based transaction behavior analysis  

---

## ⚙️ Feature Engineering

- Datetime parsing  
- Time‑difference calculation  
- One‑hot encoding of categorical variables  
- Removal of non‑modeling identifiers  
- Train/test split with stratification  

---

## 🤖 Machine Learning Models

The following models are trained and evaluated:

- Logistic Regression  
- K‑Nearest Neighbors (KNN)  
- Support Vector Classifier (SVC)  
- Random Forest  
- Decision Tree  
- Bagging Classifier  
- Voting Classifier (soft voting)

Each model outputs:

- Accuracy score  
- Classification report  
- Fraud vs. non‑fraud prediction distribution  

A sample decision tree from the Random Forest is visualized for interpretability.

---

## 📈 Results

The notebook prints:

- Accuracy for each model  
- Precision, recall, and F1‑scores  
- Fraud detection performance comparison  

These results help identify which model best captures the fraud patterns defined by the business rules.

---

## 🛠️ Requirements

Install dependencies using:

```bash
pip install pandas numpy matplotlib scikit-learn



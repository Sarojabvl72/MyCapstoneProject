# MyCapstoneProject
# 📊 EDA & Fraud Detection Notebook  
### Comprehensive Analysis of Bank Transaction Data

This repository contains a Jupyter Notebook designed to perform **Exploratory Data Analysis (EDA)** and build a **fraud detection model** using a structured dataset of bank transactions. The notebook walks through data exploration, feature engineering, fraud‑flag creation based on business rules, and multiple machine‑learning models for classification.

---

## 📁 Problem Statement

Online banking fraud has escalated due to the widespread adoption of digital channels, mobile banking, and instant payment systems. Fraudsters exploit vulnerabilities such as compromised credentials, device spoofing, synthetic identities, and rapid transaction sequences that bypass manual review. Traditional fraud detection systems—often static and rule‑based—struggle with:

- Evolving fraud patterns that change faster than rules can be updated
- High transaction velocity, making real‑time detection essential
- Massive data volumes, requiring scalable analytics
- Extreme class imbalance, where fraudulent transactions represent less than 0.5% of all activity
- Concept drift, where user behavior changes over time
- Adversarial behavior, where fraudsters intentionally mimic legitimate patterns

Machine learning and deep learning models address these challenges by learning complex patterns, adapting to new fraud behaviors, and providing real‑time detection capabilities. Modern approaches include supervised learning, anomaly detection, graph neural networks, transformer‑based architectures, and federated learning for privacy‑preserving collaboration across institutions

---
## 📁 Why Fraud Detection Models Are Critically Important

- Massive Financial Losses: 
     Financial institutions lose around 5% of annual revenue to fraud, totaling over $50 billion per year in the U.S.
- Customer Trust & Brand Reputation:  
     Fraud incidents erode customer confidence, leading to account closures and reduced digital adoption.
- Regulatory Pressure:  
      Banks must comply with strict anti‑fraud and cybersecurity regulations; failure results in fines and legal exposure.
- Operational Efficiency:  
      Automated fraud detection reduces manual review workload and false positives, lowering operational costs.
  - Real‑Time Protection:  
      ML/DL models enable instant detection and blocking of suspicious transactions before losses occur.
---

## Priority Problems to Solve

* Real‑time detection — Prevent fraudulent transactions before funds leave the account.
* Reducing false positives — Avoid unnecessary customer friction and support costs.
* Handling class imbalance — Fraud is rare; models must detect minority patterns accurately.
* Adaptive learning — Models must evolve as fraud tactics change.
* Scalability & latency — Systems must process millions of transactions per second.
* Interpretability — Regulators and analysts need transparent reasoning behind model decisions.
* Data privacy & security — Protect sensitive financial data while enabling collaborative detection.

---

## Financial Impact on Institutions

* Direct monetary losses from unauthorized transfers, account takeovers, and card‑not‑present fraud* 
* Chargeback costs and reimbursement obligations.
* Investigation and recovery expenses
* Regulatory fines for inadequate fraud controls
* Long‑term revenue loss due to customer churn and reduced digital engagement
  
- Studies show that advanced AI‑driven fraud detection systems achieve 93–96% accuracy on real transaction datasets, significantly reducing losses and improving operational resilience
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



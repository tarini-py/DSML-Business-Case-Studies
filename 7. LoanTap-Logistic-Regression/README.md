[![LinkedIn](https://img.shields.io/badge/LinkedIn-Tarini%20Prasad%20Samantray-blue?logo=linkedin&logoColor=white)](https://www.linkedin.com/in/mr-tps/)

## 🚀 Run on Google Colab

#### Spark version
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1JCVlp9h3xu_SVtbI9ACEm1EtraYFass9?usp=sharing)

## 📊 View on Kaggle

[![Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/code/tariniprasad0x/loantap-logistic-regression)

---

# LoanTap Loan Default Prediction

## 📌 Business Problem
LoanTap, a digital lending platform, needs to minimize loan defaults while continuing to lend to creditworthy customers. Incorrectly approving risky loans increases NPAs (Non-Performing Assets), while being overly strict leads to loss of good lending opportunities.

The objective of this project is to build a machine learning model that predicts loan default risk and helps optimize lending decisions.

---

## 🎯 Project Objective
- Predict whether a borrower will default or repay a loan.
- Reduce NPAs by correctly identifying risky borrowers.
- Balance recall (catching defaulters) and precision (not rejecting safe customers).
- Support data-driven credit approval decisions.

---

## 🗂 Dataset Overview
The dataset includes borrower information such as:

- Demographics
- Employment details
- Loan characteristics
- Financial history
- Credit behavior indicators

Target variable:
- Loan Status (Default / Non-Default)

---

## ⚙️ Project Workflow
1. Data cleaning and preprocessing
2. Exploratory Data Analysis (EDA)
3. Handling missing values and outliers
4. Feature engineering and encoding
5. Class imbalance handling (SMOTE / class weights)
6. Model training (Logistic Regression & variants)
7. Model evaluation using:
   - ROC-AUC
   - PR-AUC
   - Precision & Recall
   - MCC
   - Confusion Matrix
8. Threshold tuning for business optimization

---

## 📊 Evaluation Insight
- Low precision → bank loses safe lending opportunities.
- Low recall → bank faces higher NPAs.
- Threshold optimization helps balance risk and profitability.

---

## 🛠 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Imbalanced-learn

---

## 🚀 Key Outcomes
- Built a credit risk prediction pipeline.
- Reduced false approvals of risky loans.
- Improved decision-making for loan approvals.
- Demonstrated business impact through metric optimization.

---


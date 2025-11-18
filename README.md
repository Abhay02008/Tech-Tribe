# 🔮 Customer Churn Prediction  
### Machine Learning Hackathon Project

This project focuses on predicting *customer churn* using classical machine learning techniques.  
The goal is to help businesses identify customers who are likely to leave so they can take proactive retention actions.

---

## 🚀 Problem Statement

The objective is to build a *Machine Learning model* that predicts customer churn (0 = No, 1 = Yes) using demographic, behavioral, and usage-related features.

Key requirements include:

- Build and compare *Logistic Regression* and *Random Forest* models  
- Handle imbalanced data using *Class Weighting* (instead of SMOTE)  
- Evaluate model performance using:
  - Accuracy (~70%)
  - Confusion Matrix
  - ROC Curve & AUC Score

---

## 📊 Dataset Description

The dataset contains *1,000 customer records* with the following columns:

| Feature | Description |
|--------|-------------|
| Age | Customer age (18–70) |
| Gender | Male/Female |
| Tenure | Total months with company |
| Usage Frequency | Service usage (5–200) |
| Support Calls | Number of support calls |
| Payment Delay | Days payment was delayed |
| Subscription Type | Basic / Premium / Gold |
| Contract Length | Monthly / Annual / Quarterly |
| Total Spend | Total spending amount |
| Last Interaction | Days since last interaction |
| *Churn* | Target variable (0 = No, 1 = Yes) |

---
---

## 🛠 Tech Stack

- *Python 3.10.11*
- *Pandas, NumPy*
- *Scikit-Learn*
- *Matplotlib* (for plots)

---

# 🔧 Workflow Pipeline

## 1. Data Collection : 
Collect the dataset containing customer demographics, usage patterns, payment behavior, and churn labels.
**Dataset:** `customer_churn_data.csv`
## 2. Data Cleaning & Preprocessing



### 🔹 Handling Missing Values
- *Categorical features* → fill missing values with *mode*  
- *Numerical features* → fill missing values with *mean* or *median*
### 🔹 Encoding Categorical Features
Use *One-Hot Encoding* for the following features:
- *Subscription Type* (Basic, Premium, Gold)  
- *Contract Length* (Monthly, Annual, Quarterly)  
- *Gender* (Male/Female)
### 🔹 Feature Scaling
Apply *feature scaling (StandardScaler or MinMaxScaler)* for models like *Logistic Regression*, which are sensitive to feature magnitudes.
## 3. Exploratory Data Analysis (Visualizations)
-Several visualizations were created to understand feature distributions and patterns.
  - a. Distribution Plots (Numerical Features)
  - b. Churn Count Plot
  - c. Correlation Heatmap (Matplotlib)
  - d. Box Plot (Outlier Detection)

## 4. Class Imbalance Handling
- Class Weighting
- Churn = 1 is only ~70% → dataset is imbalanced.
- Why use class weighting?
    -No synthetic data needed
    -Prevents overfitting
    -orks best for small datasets (<10000 rows)
  

## 5. Model Training
- Logistic Regression
    -Logistic Regression was chosen for its simplicity, interpretability, and strong performance on small datasets. It provides a clear baseline and helps identify which features most strongly influence churn.

- Random Forest  
  -Random Forest was selected because it captures complex, non-linear patterns in customer behavior, is robust to outliers, handles mixed feature types, and provides strong accuracy with minimal tuning.

## 6. Model Evaluation
- Confusion Matrix  : Shows TP, TN, FP, FN.
- ROC Curve  : Measures performance across thresholds.
- Accuracy  : Primary metric for hackathon (~70% target).

## 7. Model Deployment
  - Streamlit (Simple UI)
  - Input fields for customer data.
  - Churn prediction output
---


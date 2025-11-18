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
- Collect the dataset of Subscribtion Based Platform  . Dataset contain training(10000 rows and 12 coloumns) and testing files .
- We imported panda as pd . then we  load the data using the pd.read_csv("dataset.csv").we give name to dataset df.
- for showing first 5 rows   we used df.head().

**Dataset:** `customer_churn_data.csv`
## 2. Data Cleaning 

- first we checked the data set is imbalanced or not , and for that i counted  the curns value  and converted into percentage using normalization .
###  checking the missing values 
  -we removed the missing churn value rows.using df.dropna(subset=["churn"]).
  -we used df.isnull().sum() to find in each column how many  null values are there.
### 🔹 Handling Missing Values
- *Categorical features* → previusly we found out  the categorical type data. and used for loop  on them and used  fillna() replace missing value (NaN) in the column using mode .
    -
- *Numerical features* →simillary we are using for loop and  used fillna()  and replace missing value (NaN) in the column  and using mean.
- we checked datatype of each attribute. and found 8attributes are float type and 3 are categorical type .
- we removed Strongs features  like Total spend, Tenure , Last Transaction so the model does not overfit  using df.drop(colmuns=strong_features,errors="ignore").
### Feture Seprating and target 
  -we are seprating all the customer information from the target using drop(). basically all the customer information is X axis and target "chrun" is Y axis . so that we can build relationship bw dataset and target .
## 3. Data Preprocessing.
🔹 Encoding Categorical Features:
we imported oneHotEncoder and StandardScaler form sklearn.preprocessing.
-Apply One-Hot Encoding to categorical columns and Standard Scaling to numerical columns using a single combined transformer.
Use *One-Hot Encoding* for the following features:
- *Subscription Type* (Basic, Premium, Gold)  
- *Contract Length* (Monthly, Annual, Quarterly)  
- *Gender* (Male/Female)
  
### 🔹 Feature Scaling
-Apply *feature scaling (StandardScaler or MinMaxScaler)* for models like *Logistic Regression*, which are sensitive to feature magnitudes.
### Train /Test Split 
  -we splits the data into 80% training and 20% testing while keeping the class proportions the same using stratify=y.

## 3. Exploratory Data Analysis (Visualizations)
-Several visualizations were created to understand feature distributions and patterns.
  - a. Distribution Plots (Numerical Features)
  - b. Churn Count Plot
  - c. Correlation Heatmap (Matplotlib)
  - d. Box Plot (Outlier Detection)



## 5. Model Training
- Logistic Regression
    -Logistic Regression was chosen for its simplicity, interpretability, and strong performance on small datasets. It provides a clear baseline and helps identify which features most strongly influence churn.
    -We are importing Logistic Regression model from  sklearn .
    -I created a Pipeline that combines preprocessing and logistic regression into a single workflow.LogisticRegression is applied with max_iter=200.
  -This Pipeline ensures that the exact same preprocessing is applied during both training and testing, prevents data leakage, and makes the model easier to maintain and deploy.
  -After that , I train the model using log_reg_model.fit(X_train,Y_train) .
  -After that , I made prediction using Testing data by using log_reg_model.predict(X_test).
  -Finally i checked the Accuracy_score(y_test,y_pred). we got accuracy  of 72%
  
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


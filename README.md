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
## 1. Data Collection

- Collected the dataset from a **Subscription-Based Platform**. The dataset contains training (10,000 rows and 12 columns) and testing files.  
- Imported `pandas` as `pd` and loaded the data using `pd.read_csv("customer_churn_data.csv")`. The dataset was assigned to the variable `df`.  
- Displayed the first five rows using `df.head()`.

**Dataset:** `customer_churn_data.csv`

---

## 2. Data Cleaning

- Checked whether the dataset is imbalanced by counting the churn values and converting them into percentages using normalization.

### Checking Missing Values
- Removed rows with missing churn values using `df.dropna(subset=["churn"])`.  
- Used `df.isnull().sum()` to find the number of null values in each column.

### Handling Missing Values
- **Categorical features:** Identified categorical columns and used a `for` loop with `fillna()` to replace missing (NaN) values with the mode.  
- **Numerical features:** Similarly, used a `for` loop with `fillna()` to replace missing (NaN) values with the mean.

- Checked the data types and found 8 numerical (float) attributes and 3 categorical attributes.  
- Removed strong features like `Total_Spend`, `Tenure`, and `Last_Transaction` using `df.drop(columns=strong_features, errors="ignore")` to prevent model overfitting.

### Feature Separation and Target
- Separated the customer information (features) from the target variable.  
- Used `drop()` to create `X` (features) and `y` (target), where `churn` is the target variable. This separation helps in building relationships between features and the target.

---

## 3. Data Preprocessing

### Encoding Categorical Features
- Imported `OneHotEncoder` and `StandardScaler` from `sklearn.preprocessing`.  
- Applied **One-Hot Encoding** to categorical columns and **Standard Scaling** to numerical columns using a combined transformer.

**One-Hot Encoded Features:**
- Subscription Type (Basic, Premium, Gold)  
- Contract Length (Monthly, Annual, Quarterly)  
- Gender (Male, Female)

### Feature Scaling
- Applied feature scaling using `StandardScaler` (or `MinMaxScaler`) for models like Logistic Regression, which are sensitive to magnitude differences.

### Train/Test Split
- Split the dataset into 80% training and 20% testing, maintaining class proportions using `stratify=y`.

---

## 4. Exploratory Data Analysis (EDA)

Created several visualizations to understand feature distributions and patterns:

- Distribution plots (for numerical features)  
- Churn count plot  
- Correlation heatmap (using Matplotlib)  
- Box plot (for outlier detection)

---

## 5. Model Training

- **Model Used:** Logistic Regression  
- Chosen for its simplicity, interpretability, and efficiency on small datasets. It also provides clear insight into which features influence churn.

### Steps:
1. Imported `LogisticRegression` from `sklearn.linear_model`.  
2. Created a `Pipeline` combining preprocessing steps and the model for consistent transformation during both training and testing.  
3. Configured the model with `max_iter=200`.  
4. Trained the model using `log_reg_model.fit(X_train, y_train)`.  
5. Made predictions on the test set using `log_reg_model.predict(X_test)`.  
6. Evaluated performance using `accuracy_score(y_test, y_pred)`.

**Result:** Achieved an accuracy of approximately **72%**.

---

## 5. Model Training

### Logistic Regression

- **Model Used:** Logistic Regression  
- Chosen for its simplicity, interpretability, and strong performance on small datasets. This model helps identify which features most influence customer churn.

#### Steps:
1. Imported `LogisticRegression` from `sklearn.linear_model`.
2. Created a `Pipeline` that combines preprocessing steps with the logistic regression model, ensuring consistent transformations during both training and testing.
3. Configured the model with `max_iter=200` to allow sufficient iterations for convergence.
4. Trained the model with `log_reg_model.fit(X_train, y_train)`.
5. Predicted labels on the test set using `log_reg_model.predict(X_test)`.
6. Evaluated model performance using `accuracy_score(y_test, y_pred)`.

**Result:** Achieved an accuracy of approximately **72%**.

---

### Random Forest

- **Model Used:** Random Forest Classifier  
- Chosen for its robustness to overfitting and ability to capture complex, non-linear relationships in the data.

#### Steps:
1. Imported `RandomForestClassifier` from `sklearn.ensemble`.
2. Used `Pipeline` from `sklearn.pipeline` to combine data preprocessing and model training.
3. Configured the classifier with `n_estimators=200` (number of trees), `max_depth=None` (full depth), and `random_state=42` (reproducibility).
4. Trained the pipeline using `rf_model.fit(X_train, y_train)`.
5. Predicted test set labels with `rf_model.predict(X_test)`.
6. Evaluated the model with `accuracy_score(y_test, rf_pred)` and `classification_report(y_test, rf_pred)`.

**Result:** Achieved an accuracy of approximately **71%**.

---



## 6. ROC & AUC Curves

Used ROC curves and AUC scores to assess the classification performance and discriminative ability of both Logistic Regression and Random Forest models.

---

### Import Libraries

- Imported `roc_auc_score` and `roc_curve` from `sklearn.metrics` for ROC curve analysis.
- Imported `matplotlib.pyplot` for plotting the ROC curves.

---

### 6.1 — Logistic Regression ROC–AUC

- Generated predicted probabilities for the positive class using `predict_proba`.
- Calculated the AUC (Area Under the Curve) score for Logistic Regression.

---

### 6.2 — Random Forest ROC–AUC

- Generated predicted probabilities for the positive class using `predict_proba`.
- Calculated the AUC score for Random Forest.

---

### 6.3 — Plot ROC Curves

- Computed the False Positive Rate (FPR) and True Positive Rate (TPR) for both models.
- Plotted ROC curves to compare model performance visually.




## 7. Model Deployment
  - Streamlit (Simple UI)
  - Input fields for customer data.
  - Churn prediction output
---


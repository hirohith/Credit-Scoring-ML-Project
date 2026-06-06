# Credit Scoring Model Using Machine Learning

## Project Overview

This project focuses on building a Machine Learning-based Credit Scoring System that predicts whether a customer is **creditworthy** or **risky** using financial and demographic information.

Banks and financial institutions use credit scoring systems to reduce loan default risk and improve financial decision-making.

The project uses Machine Learning classification algorithms along with data preprocessing and class balancing techniques to improve prediction accuracy and risky-customer detection.

---

## Problem Statement

Traditional credit evaluation systems are:

- Time-consuming
- Inconsistent
- Difficult to scale for large datasets

Incorrect loan approvals can lead to:

- Financial losses
- Increased loan defaults
- Inefficient banking operations

This project aims to automate customer credit risk prediction using Machine Learning techniques.

---

## Project Objectives

- Build a predictive credit scoring system
- Handle missing and categorical data
- Train and compare Machine Learning models
- Improve risky-customer detection
- Analyze important financial risk factors
- Evaluate models using multiple performance metrics

---

## Dataset Used

### German Credit Dataset

The dataset contains customer financial and demographic details.

### Features Used

- Age
- Sex
- Job
- Housing
- Saving Accounts
- Checking Account
- Credit Amount
- Duration
- Purpose

### Target Variable

**Risk (Good / Bad)**

### Dataset Size

| Attribute | Value |
|------------|--------|
| Records | 1000 |
| Features | 9 |

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- imbalanced-learn
- Streamlit
- Joblib

---

## Project Workflow

```text
Dataset Collection
        ↓
Data Cleaning
        ↓
Missing Value Handling
        ↓
Categorical Encoding
        ↓
Feature Scaling
        ↓
Train-Test Split
        ↓
Model Training
        ↓
SMOTE Balancing
        ↓
Performance Evaluation
        ↓
Model Deployment
```

---

## Data Preprocessing

### Missing Value Handling

Missing values in:

- Saving Accounts
- Checking Account

were replaced using:

```python
df.fillna("unknown", inplace=True)
```

### Label Encoding

Categorical features were converted into numerical values.

Example:

```text
male   → 1
female → 0

good   → 1
bad    → 0
```

### Feature Scaling

Feature scaling was performed using:

```python
StandardScaler()
```

This normalized feature ranges and improved model learning performance.

---

## Train-Test Split

The dataset was divided into:

- 80% Training Data
- 20% Testing Data

```python
train_test_split(test_size=0.2, random_state=42)
```

---

## Machine Learning Models Used

### Logistic Regression

Used as the baseline classification model.

**Advantages**

- Fast
- Simple
- Interpretable

### Random Forest Classifier

Used for improved nonlinear pattern detection and better prediction stability.

**Advantages**

- Reduces overfitting
- Better generalization
- Higher predictive performance

---

## Handling Class Imbalance Using SMOTE

The dataset contained more good customers than risky customers.

This caused biased predictions.

To solve this issue, **SMOTE (Synthetic Minority Oversampling Technique)** was applied to balance the training dataset.

```python
SMOTE(random_state=42)
```

---

## Model Evaluation Metrics

The following metrics were used:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- ROC-AUC Curve

---

## Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|---------|----------|-----------|---------|----------|
| Logistic Regression | 75.5% | 77.7% | 91.5% | 84.0% |
| Random Forest | 76.5% | 79.0% | 90.8% | 84.5% |
| Random Forest + SMOTE | 76.0% | 82.9% | 82.9% | 82.9% |

---

## Key Findings

- Random Forest performed better than Logistic Regression.
- SMOTE significantly improved risky-customer detection.
- Credit Amount, Age, and Duration were the most influential features.
- Financial behavior had greater impact than demographic features.

---

## Visualizations Included

- Confusion Matrix
- ROC Curve
- Feature Importance Graph

---

## Streamlit Web Application

A simple Streamlit web application was developed for real-time customer credit risk prediction.

Users can:

- Enter customer financial details
- Predict customer creditworthiness
- Analyze risk classification

Run the application:

```bash
streamlit run app.py
```

---

## Advantages of the System

- Automated financial risk analysis
- Faster loan approval decisions
- Reduced human error
- Improved risky-customer identification
- Scalable Machine Learning solution

---

## Limitations

- Limited dataset size
- Class imbalance issue
- Synthetic oversampling may introduce slight bias
- Real-world banking systems use more complex datasets

---

## Future Scope

- Implement XGBoost algorithm
- Use larger real-world banking datasets
- Cloud deployment
- Real-time banking integration
- Advanced financial fraud detection
- Interactive dashboard improvements

---

## Conclusion

This project successfully developed a Machine Learning-based Credit Scoring System capable of predicting customer financial risk using financial and demographic data.

Among all models, **Random Forest combined with SMOTE** provided the best balanced performance and improved risky-customer detection.

The project demonstrated the importance of preprocessing, class balancing, and evaluation metrics in building reliable financial prediction systems.

---

## How to Run the Project

### Clone Repository

```bash
git clone <repository-url>
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Jupyter Notebook

```bash
jupyter notebook
```

### Run Streamlit App

```bash
streamlit run app.py
```

---

## Author

**Sheru Rohith**  
B.Tech CSE, SAGE University  
Machine Learning & Data Analytics Enthusiast

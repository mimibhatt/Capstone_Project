# Predicting credit risk using Machine Learning
In this project I am comparing various classification models to predict if there is a risk of default with a credit. 
Jupiter notebook with code, experimentation, implementation is located [here]

# Business Goals
This project aims to build a machine learning model that predicts the likelihood of a borrower defaulting on a loan. The model will help lending institutions assess the risk of potential borrowers and make informed lending decisions. By analyzing borrower data and their financial history, the model will classify borrowers into two categories:

Default: The borrower is likely to default on the loan.

Non-default: The borrower is likely to repay the loan.

This tool can assist in reducing loan defaults, improving loan approval decisions, and optimizing financial risk management.

# Data
The data set used is available [here](https://www.kaggle.com/datasets/bytadit/bank-loan-dataset-2014-2017).  It contains lending information for borrowers of different credit plans and whether the borrower paid or defaulted.\
The original data set has 466285 rows and 75 columns. Using 5% of the data brought it down to 23314 rows and 75 columns. After data cleansing and feature engineering, the final data set contained 11971 rows and 21 columns.


# Workflow
Exploratory Data Analysis (EDA): Initial data exploration to understand patterns, correlations, and data quality. Visualizations and summary statistics are used to gain insights into the relationships between features.

## Data Preprocessing:

Handling missing values\
Removing categorical columns with too many unique values\
Removing columns with more than 50% null values\
Removing column with extremely unbalanced data\
Encoding categorical variables (e.g., one-hot encoding)\
Scaling numeric features (e.g., standardization)\
Balancing the dataset if necessary (e.g., using SMOTE for imbalanced classes)

## Feature Engineering:

Creating new features based on existing ones (e.g.- the binary target variable was created using the loan_status column)\
Selecting first level of features based on the correlation matrix\
Selecting important features using techniques like Recursive Feature Elimination (RFE) after the baseline modelling

## Model Training:

Training multiple machine learning models on the training dataset.
Using cross-validation to ensure robustness.
Fine-tuning hyperparameters using GridSearchCV to optimize performance.

## Model Evaluation:

Models are evaluated using metrics like accuracy, precision, recall, F1-score, and AUC-ROC to assess how well the model is predicting loan defaults.
Confusion matrices and ROC curves are used to interpret model performance.

## Model Selection:

The best-performing model is selected based on evaluation metrics.
Feature importance and interpretability are taken into consideration.

# Modeling and Performance

I used various classification models for a baseline comparison. Below are the metrics

<img width="420" alt="image" src="https://github.com/user-attachments/assets/53fd9ed6-a005-4bb7-9a35-4f3e05b80a6b">

<img width="686" alt="image" src="https://github.com/user-attachments/assets/a2c43bc5-f8d3-4558-b61f-8d7efb07d745">


# Conclusion
From the baseline comparison, DecisionTree Classifier seems to give the highest score in the shortest processing time. SVM took a long time, hence had to process using 10% random sample.

# Predicting credit risk using Machine Learning
In this project I am comparing various classification models to predict if there is a risk of default with a credit. 
Jupiter notebook with code, experimentation, implementation is located [here](https://github.com/mimibhatt/Capstone_Project/blob/main/loan_analysis.ipynb)

# Business Goals
This project aims to build a machine learning model that predicts the likelihood of a borrower defaulting on a loan. The model will help lending institutions assess the risk of potential borrowers and make informed lending decisions. By analyzing borrower data and their financial history, the model will classify borrowers into two categories:

Default: The borrower is likely to default on the loan.

Non-default: The borrower is likely to repay the loan.

This tool can assist in reducing loan defaults, improving loan approval decisions, and optimizing financial risk management.

# Data
The data set used is available [here](https://github.com/mimibhatt/Capstone_Project/tree/main/Data).  It contains lending information for borrowers of different credit plans and whether the borrower paid or defaulted.\
The original data set has 466285 rows and 75 columns. Due to computational challenges, using 5% of the original data to being with which is 23314 rows and 75 columns. After data cleansing and feature engineering, the final data set contained 11971 rows and 21 columns.


# Workflow
## Exploratory Data Analysis (EDA):

Initial data exploration to understand patterns, correlations, and data quality. Below is a visualizations and summary statistics of how each feature affects the target variable or final status of good or bad loan.\
<img width="1034" alt="Screenshot 2024-09-30 085225" src="https://github.com/user-attachments/assets/093f0b42-229f-40a3-a964-4fdbc07af6fc">


## Data Preprocessing:
The following data processing and cleanup performed on the data set
Handling missing values\
Removing categorical columns with too many unique values\
Removing columns with more than 50% null values\
Removing column with extremely unbalanced data\
Encoding categorical variables (e.g., one-hot encoding)\
Scaling numeric features (e.g., standardization)\
Balancing the dataset if necessary (e.g., using SMOTE for imbalanced classes)

## Feature Engineering:

Some categorization was done to reduce unique values\
Creating new features based on existing ones (e.g.- the binary target variable was created using the loan_status column)\
Selecting first level of features based on the correlation matrix\
Selecting important features using techniques like Recursive Feature Elimination (RFE) after the baseline modelling

# Modeling

## Model Training:
The following classification models were used for baseline assessment:
LogisticRegression, RandomForestClassifier, SVM, DecisionTreeClassifier, KNeighborsClassifier
After baseline, fine-tuning hyperparameters was done using GridSearchCV to optimize performance.

## Model Evaluation:

### Modeling Metrics and Performance

I used various classification models for a baseline comparison. For this business case I am choosing to evaluate 2 metrics: 
### Recall and Precision
A high Recall (True positive rate) would mean the model is able to correctly identify the borrowers who are likely to default. This is very important for a lending company as missing a borrower who will default may result in significant loss for the institution.
The other important metric is Precision, which can measure the proportion of borrowers likely to default who actually default. A high precision would minimize false positives. Rejecting a loan to a non-defaulter may be costly for a lending company or damage customer relationship, hence this metric is important.

So overall, to get a balance between Precision and Recall, I am going for a higher F1 score. Random Forest Classifier seems to give the best result.

<img width="731" alt="Screenshot 2024-09-30 060733" src="https://github.com/user-attachments/assets/4a4a5a3e-b17b-499e-8cd3-49b31aa5cd6d">

<img width="406" alt="Screenshot 2024-09-30 061323" src="https://github.com/user-attachments/assets/9dc5b4f6-e840-4ad5-a024-dc25ad31f9ac">

#### After Hyperparameter tuning with GridSearchCV

<img width="404" alt="Screenshot 2024-09-30 063516" src="https://github.com/user-attachments/assets/34328edb-490b-4f0b-9565-0a3be42dac66">


# Findings
The key features influencing the credit risk are:\
Loan amount: Total amount of the loan\
Annual income: Borrower’s annual income\
Debt-to-Income ratio (DTI): Ratio of borrower’s monthly debt payments to monthly income\
Employment length: Number of years the borrower has been employed\
Credit score: Borrower’s credit rating\
Purpose of loan: Reason for taking the loan\
Payment history: Borrower’s payment behavior on past loans

# Recommendations
## Model Deployment
RandomForestClassifier seems to give the best result with respect to Precision and Recall. Since the model seems to be performing at > 90% accuracy, it is production ready and ready to be deployed
## Feature Monitoring
Debt-to-income ratio, credit score, and payment history are the most critical factors in determining loan default risk. It is recommended that the company closely monitor these factors when approving loans
## Handling Imbalanced Data
Since default loans are less frequent, using techniques like SMOTE to handle class imbalance in future datasets is recommended if the model performance decreases

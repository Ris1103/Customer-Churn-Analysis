# Customer Churn Prediction

This project aims to predict customer churn using various machine learning models. Churn prediction helps businesses retain customers by identifying those at risk of leaving. The project evaluates multiple models, compares their performance, and identifies the best model for predicting customer churn.

## Dataset

The dataset used in this project is a **Customer Churn** dataset available from this [source](https://raw.githubusercontent.com/anilak1978/customer_churn/master/Churn_Modeling.csv).

The dataset contains the following features:

- **CreditScore**: The customer's credit score.
- **Geography**: The country the customer is located in.
- **Gender**: The customer's gender.
- **Age**: The customer's age.
- **Tenure**: The number of years the customer has been with the bank.
- **Balance**: The customer's account balance.
- **NumOfProducts**: The number of products the customer uses.
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No).
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No).
- **EstimatedSalary**: The customer's estimated salary.
- **Exited**: Whether the customer churned (1 = Yes, 0 = No) - this is the target variable.

## Project Overview

In this project, we perform the following steps:

1. **Exploratory Data Analysis (EDA)**:

   - A thorough EDA to understand the data distribution, relationships between features, and correlations.
   - Visualizations such as correlation heatmaps, distribution plots, boxplots, and pair plots are used to gain insights into the data.
2. **Data Preprocessing**:

   - Handling missing values (though none were found in this dataset).
   - Encoding categorical variables (e.g., Geography, Gender).
   - Scaling features where necessary, especially for models like Logistic Regression and SVM.
3. **Modeling**:

   - Several machine learning algorithms were implemented, including:
     - Logistic Regression
     - Decision Tree
     - Random Forest
     - Support Vector Machine (SVM)
     - Gradient Boosting
     - XGBoost (with and without scaling)
   - These models were compared using metrics such as Accuracy, Precision, Recall, F1 Score, and ROC-AUC.
4. **Model Evaluation**:

   - The models were compared based on their performance in predicting customer churn.
   - The best-performing model, **Random Forest**, was chosen based on its **F1 Score**, which balances **Precision** and **Recall**, and    its high **Recall** value, making it effective at identifying customers who are at risk of churning.

## Conclusion

From the model evaluation, the following conclusions were drawn:

1. **Random Forest** achieved the highest performance across most metrics, especially in **Precision**, **Recall**, **F1 Score**, and **Accuracy**.
   - It was chosen as the best model based on its **F1 Score** and **Recall**, which are important in churn prediction scenarios where both false positives and false negatives have a business impact.
2. **XGBoost with and without Scaling** performed identically and was a close second to Random Forest. It provides strong generalization, as evidenced by its **ROC-AUC** and **F1 Score**.
3. **Gradient Boosting** was consistent but slightly outperformed by both **Random Forest** and **XGBoost**.
4. **Decision Tree** performed well in **Recall** but fell short in other metrics like **F1 Score** and **Precision**.
5. **Logistic Regression**, even after scaling, was the weakest model, with low **Recall** and **F1 Score**, making it unsuitable for this task.

## Visualizations

Various visualizations were created to gain insights into the dataset:

- **Correlation heatmap** to explore relationships between features.
- **Distribution plots** for continuous features like age, credit score, and balance.
- **Boxplots** and **Count plots** for churn versus categorical and numerical features.
- **Pair plots** to explore feature interactions.

### Prerequisites

- Python 3.x
- Jupyter Notebook
- The following Python libraries:
  - pandas
  - numpy
  - seaborn
  - matplotlib
  - scikit-learn
  - xgboost

## Best Model Criteria

- **Random Forest** was selected as the best model based on its **F1 Score**, which balances **Precision** and **Recall**. This makes it ideal for churn prediction, where both false positives (predicting churn when it won’t happen) and false negatives (failing to identify a churner) can have business implications.
- **Recall** is another critical metric, and **Random Forest** had the highest recall, meaning it captures the most actual churn cases.

## Future Improvements

- **Hyperparameter tuning**: Further optimize models using GridSearchCV or RandomizedSearchCV.
- **Feature engineering**: Create new features or transform existing ones to improve model performance.
- **Ensemble techniques**: Combine multiple models (stacking, blending) to improve accuracy.

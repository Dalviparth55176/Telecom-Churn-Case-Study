# Telecom-Churn-Case-Study
Analysis and prediction on customer churn

Telecom Churn Prediction - Machine Learning Model
Overview
This project focuses on predicting customer churn in the telecommunications industry. The goal is to identify high-value customers who are at risk of leaving their current telecom provider. By using machine learning, we aim to predict churn and assist in targeting retention strategies to reduce customer loss, especially among high-value users.

In a highly competitive telecom market, where churn rates average 15-25% annually, retaining existing customers is crucial. It costs 5-10 times more to acquire a new customer than to retain an existing one. This makes customer retention a key business priority.
This project is based on the Indian and Southeast Asian market.

**Churn Definition**
Types of Churn:
**Revenue-based Churn:** Customers who have not utilized any revenue-generating services (calls, SMS, mobile internet) for a defined period.
One could also use aggregate metrics such as ‘customers who have generated less than INR 4 per month in total/average/median revenue.

**Shortcoming:** Some customers may only receive calls/SMS, generating no revenue but still using the service.
Late-stage churn prediction. If a customer has already stopped using services for a while, it may be too late to take corrective actions.

**Usage-based Churn:** Customers who have not made any usage, whether incoming or outgoing, over a defined period.

**High-Value Customers:**
In the Indian and Southeast Asian markets, around 80% of revenue comes from 20% of customers. These customers are considered high-value, and reducing churn among them can significantly reduce revenue leakage.

In this project, churn is defined usage-based and focuses on predicting churn among high-value customers only.

**Objective**
The goal of this project is to:

Build a predictive model that identifies high-value customers at risk of churn based on their usage patterns.
Analyze key features and identify indicators that correlate with customer churn.
Propose strategies for telecom providers to retain high-value customers.

**Dataset**

The dataset contains customer-level information for a span of four consecutive months - June, July, August and September. The months are encoded as 6, 7, 8 and 9, respectively. ch as:

Churn information (whether the customer has churned)
Customers usually do not decide to switch to another competitor instantly, but rather over a period of time (this is especially applicable to high-value customers). In churn prediction, we assume that there are three phases of the customer lifecycle :

**The ‘good’ phase:** In this phase, the customer is happy with the service and behaves as usual.

**The action phase:** The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than in the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)

**The ‘churn’ phase:** In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.

**Data Preprocessing:** 
Handling missing values: Dropping varaibels with more than30% missing values, columns with one unique values rows with 50 % null values.  
Removing outliers through IQR method 
Filtering high value customers with 70th percentile of the average recharge amount in the first two months (the good phase).
**Steps involved in the churn prediction process**:
Data Exploration: Understanding the features and distributions, visualizing the data to identify patterns.
Feature Engineering: Creating new features based on the existing ones (e.g., minutes of uasege, decrease in recharge amount etc).
Treating imbalanced data with Synthetic Minority Over-sampling Technique (SMOTE).
Model Selection: Using the following machine learning models to predict churn:
**Logistic Regression:** A simple yet effective model for binary classification.
**Decision Tree Classifier:** A non-linear classifier that provides more flexibility and interpretable decision-making for predicting churn.
Model Evaluation: Using performance metrics such as accuracy, precision, recall, F1-score, and ROC-AUC to evaluate models.
Feature Importance: Identifying the most important features that influence churn prediction.

**Final Model:**
Both the Logistic Regression and Decision Tree Classifier models were evaluated, with the goal of predicting whether high-value customers are likely to churn in the near future based on their usage and account characteristics.


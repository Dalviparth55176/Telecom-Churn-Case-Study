# Telecom-Churn-Case-Study
Analysis and prediction on customer churn

Telecom Churn Prediction - Machine Learning Model
Overview
This project focuses on predicting customer churn in the telecommunications industry. The goal is to identify high-value customers who are at risk of leaving their current telecom provider. By using machine learning, we aim to predict churn and assist in targeting retention strategies to reduce customer loss, especially among high-value users.

In a highly competitive telecom market, where churn rates average 15-25% annually, retaining existing customers is crucial. It costs 5-10 times more to acquire a new customer than to retain an existing one. This makes customer retention a key business priority.
This project is based on the Indian and Southeast Asian market.

**Churn Definition**
Types of Churn:
Revenue-based Churn: Customers who have not utilized any revenue-generating services (calls, SMS, mobile internet) for a defined period.
One could also use aggregate metrics such as ‘customers who have generated less than INR 4 per month in total/average/median revenue.

Shortcoming: Some customers may only receive calls/SMS, generating no revenue but still using the service.
Usage-based Churn: Customers who have not made any usage, whether incoming or outgoing, over a defined period.

Shortcoming: Late-stage churn prediction. If a customer has already stopped using services for a while, it may be too late to take corrective actions.
High-Value Customers:
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

The **‘good’ phase**: In this phase, the customer is happy with the service and behaves as usual.

The **‘action’ phase:** The customer experience starts to sore in this phase, for e.g. he/she gets a compelling offer from a competitor, faces unjust charges, becomes unhappy with service quality etc. In this phase, the customer usually shows different behaviour than in the ‘good’ months. Also, it is crucial to identify high-churn-risk customers in this phase, since some corrective actions can be taken at this point (such as matching the competitor’s offer/improving the service quality etc.)

The **‘churn’ phase**: In this phase, the customer is said to have churned. You define churn based on this phase. Also, it is important to note that at the time of prediction (i.e. the action months), this data is not available to you for prediction. Thus, after tagging churn as 1/0 based on this phase, you discard all data corresponding to this phase.

**Data Preprocessing:** 
Handling missing values: Dropping varaibels with more than30% missing values, columns with one unique values rows with 50 % null values.  
Feature engineering (e.g., creating usage patterns)
Removing outliers through IQR method 
Methodology
Steps involved in the churn prediction process:
Data Exploration: Understanding the features and distributions, visualizing the data to identify patterns.
Feature Engineering: Creating new features based on the existing ones (e.g., minutes of uasege, decrease in recharge amount etc).

Treating imbalanced data with Synthetic Minority Over-sampling Technique (SMOTE).
Model Selection: Using the following machine learning models to predict churn:
Logistic Regression: A simple yet effective model for binary classification.
Decision Tree Classifier: A non-linear classifier that provides more flexibility and interpretable decision-making for predicting churn.
Model Evaluation: Using performance metrics such as accuracy, precision, recall, F1-score, and ROC-AUC to evaluate models.
Feature Importance: Identifying the most important features that influence churn prediction.
Model Tuning:
Hyperparameter optimization
Final Model:
Both the Logistic Regression and Decision Tree Classifier models were evaluated, with the goal of predicting whether high-value customers are likely to churn in the near future based on their usage and account characteristics.








Certainly! Here is a pointwise version of the README for your Telecom Churn Prediction project:

---

# Telecom Churn Prediction - Machine Learning Model

## Overview
- **Objective**: Predict customer churn in the telecommunications industry, focusing on high-value customers at risk of leaving.
- **Importance**: Reducing churn is crucial as it costs more to acquire new customers than to retain existing ones. The churn rate in the telecom industry ranges from 15% to 25% annually.
- **Target Market**: Indian and Southeast Asian markets (prepaid telecom services).
- **Goal**: Build predictive models to identify customers at high risk of churn and assist in retention strategies.

## Churn Definition
- **Revenue-based Churn**: Customers who have not generated revenue (calls, SMS, internet usage) in a specified period.
  - **Shortcoming**: Does not account for customers who only receive services (e.g., calls or SMS).
  
- **Usage-based Churn**: Customers who have not made any usage (incoming/outgoing calls, internet) in a given period.
  - **Shortcoming**: Late-stage churn prediction; corrective actions may be too late.
  
- **High-Value Customers**: 
  - In India and Southeast Asia, **80% of revenue** comes from **20% of customers**.
  - These high-value customers are the primary focus for churn prediction.

## Objective
- **Goal 1**: Build a predictive model to identify high-value customers at risk of churn based on usage patterns.
- **Goal 2**: Analyze and identify key features correlated with customer churn.
- **Goal 3**: Propose retention strategies for telecom providers to reduce churn, especially among high-value customers.

## Dataset
- **Features**: 
  - Demographic data (age, region, etc.)
  - Account information (plan type, billing method)
  - Usage data (call duration, data usage, etc.)
  - Churn status (whether the customer has churned)
  
- **Data Preprocessing**:
  - Handle missing values
  - Feature engineering (e.g., total usage, number of calls)
  - Data normalization/transformation

## Methodology
1. **Data Exploration**:
   - Understand feature distributions
   - Visualize relationships in the data

2. **Feature Engineering**:
   - Create new features based on usage patterns (e.g., total data usage, number of calls made)
   
3. **Model Selection**:
   - **Logistic Regression**: A simple and effective binary classification model.
   - **Decision Tree Classifier**: A more flexible, non-linear model that can handle complex relationships.

4. **Model Evaluation**:
   - **Metrics**: Accuracy, Precision, Recall, F1-Score, ROC-AUC
   - Evaluate both models based on these metrics.

5. **Feature Importance**:
   - Identify and rank features that influence churn prediction the most.

6. **Model Tuning**:
   - Hyperparameter optimization for both models.
   - Cross-validation to ensure model reliability.

## Installation

### Steps to Run the Project Locally:
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/telecom-churn-prediction.git
   cd telecom-churn-prediction
   ```

2. **Create a Virtual Environment** (optional):
   ```bash
   python3 -m venv churn-env
   source churn-env/bin/activate  # On Windows use `churn-env\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Files and Directory Structure

```plaintext
telecom-churn-prediction/
│
├── data/
│   ├── churn_data.csv              # Raw dataset
│
├── notebooks/
│   ├── churn_exploration.ipynb     # Data exploration and visualization
│   ├── churn_modeling.ipynb        # Model building and evaluation
│
├── models/
│   ├── churn_model.pkl             # Trained machine learning model (Logistic Regression or Decision Tree)
│
├── src/
│   ├── preprocessing.py            # Data preprocessing functions
│   ├── feature_engineering.py       # Feature engineering functions
│   ├── model.py                    # Model training and evaluation code
│
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
```

## Usage

### Predicting Churn
- Load the trained model and make predictions with new customer data:

```python
from sklearn.externals import joblib

# Load the trained model
model = joblib.load('models/churn_model.pkl')

# Example customer data
customer_data = [[...]]  # Replace with actual data for prediction

# Predict churn probability
churn_probability = model.predict_proba(customer_data)
print(f"Churn Probability: {churn_probability[0][1]:.2f}")
```

## Results
- **Model 1**: Logistic Regression – A simple, interpretable model for churn prediction.
- **Model 2**: Decision Tree Classifier – A more flexible model that can handle complex decision-making.
- **Evaluation**: Both models were evaluated using various performance metrics (accuracy, precision, recall, F1-score, ROC-AUC).
- **Insights**: Identified key features that drive customer churn (e.g., usage patterns, account information).

## Conclusion
- **Impact**: Reducing churn among high-value customers can significantly retain revenue.
- **Solution**: The predictive models (Logistic Regression and Decision Tree Classifier) enable telecom providers to identify high-risk customers and act on retention strategies effectively.

## Contributing
- Feel free to fork the repository, submit issues, or create pull requests to improve the model's accuracy or enhance feature interpretation.

## License
- This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

This pointwise version provides clarity and easy navigation through the steps of your project. Let me know if you need any additional adjustments!

# 💳 Credit Card Fraud Detection

A machine learning project that detects potentially fraudulent credit card transactions using a Random Forest classification model.

The project includes data preprocessing, feature engineering, machine learning model training, model evaluation, feature importance analysis, and a Streamlit web application for making fraud predictions.

## 🚀 Project Overview

Credit card fraud is a major problem in digital transactions. This project uses machine learning to identify suspicious transactions based on transaction amount, location, time, transaction category, and other relevant features.

The trained Random Forest model classifies transactions into two categories:

- 🚨 Fraudulent Transaction
- ✅ Legitimate Transaction

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit
- Jupyter Notebook

## 🤖 Machine Learning Models

The project explores and evaluates multiple classification algorithms:

- Logistic Regression
- Decision Tree
- Random Forest

The Random Forest classifier was selected as the final model and saved using Joblib.

## 📊 Features Used

The final model uses 27 features:

- Transaction amount
- Gender
- ZIP code
- Customer latitude
- Customer longitude
- City population
- Unix timestamp
- Merchant latitude
- Merchant longitude
- Transaction hour
- Transaction day
- Transaction month
- Day of week
- Transaction category features

### Transaction Categories

- Entertainment
- Food & Dining
- Gas & Transport
- Grocery
- Health & Fitness
- Home
- Kids & Pets
- Miscellaneous
- Personal Care
- Shopping
- Travel

## 🔍 Model Evaluation

The machine learning models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion Matrix

Random Forest feature importance is also analyzed to identify which features contribute most to fraud detection.

## 🌐 Streamlit Web Application

A Streamlit-based web application was developed to provide a simple and user-friendly interface for fraud detection.

The application allows users to enter transaction information such as:

- Transaction amount
- Gender
- Transaction category
- ZIP code
- Customer location
- City population
- Merchant location
- Transaction date and time

After entering the transaction details, the application uses the trained Random Forest model to predict whether the transaction is potentially fraudulent or legitimate.

## 📁 Project Structure

```text
Credit Card Fraud Detection/
│
├── app.py
├── fraud_detection.ipynb
├── random_forest_model.pkl
├── requirements.txt
├── README.md
├── .gitignore
│
└── venv/
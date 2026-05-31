# Predictive-Modeling-and-Risk-Scoring-for-Bank-Customer-Churn
An end-to-end machine learning project that predicts bank customer churn using predictive analytics, feature engineering, and risk scoring. Includes explainable AI, churn probability prediction, model comparison, and an interactive Streamlit dashboard for proactive customer retention strategies.


# Predictive Modeling and Risk Scoring for Bank Customer Churn

## Project Overview

This project focuses on predicting customer churn in banking systems using machine learning and risk scoring techniques. The system identifies customers who are likely to leave the bank and assigns churn probability scores to help banks take proactive retention actions.

The project combines:

- Predictive analytics
- Machine learning
- Risk scoring
- Explainable AI
- Interactive dashboarding

------------------------------------------------------------

## Problem Statement

Banks often lose customers without early warning systems to detect churn risk. Traditional churn analysis is reactive and identifies churn only after customers leave.

This project builds a predictive churn intelligence system that:

- Predicts customer churn
- Generates churn probability scores
- Identifies major churn drivers
- Helps banks improve customer retention strategies

------------------------------------------------------------

## Objectives

### Primary Objectives

- Predict customer churn accurately
- Generate churn probability scores
- Identify key factors influencing churn

### Secondary Objectives

- Reduce false churn predictions
- Improve model interpretability
- Enable scenario-based churn analysis

------------------------------------------------------------

## Dataset Description

The dataset contains customer-level banking information.

### Important Features

- CreditScore
- Geography
- Gender
- Age
- Tenure
- Balance
- NumOfProducts
- HasCrCard
- IsActiveMember
- EstimatedSalary

### Target Variable

Exited

Where:

0 → Customer retained
1 → Customer churned

------------------------------------------------------------

## Project Workflow

Dataset Collection
↓
Exploratory Data Analysis
↓
Data Preprocessing
↓
Feature Engineering
↓
Train-Test Split
↓
Machine Learning Models
↓
Model Evaluation
↓
Explainability Analysis
↓
Risk Scoring
↓
Streamlit Dashboard

------------------------------------------------------------

## Feature Engineering

The following engineered features are created:

- BalanceSalaryRatio
- ProductEngagement
- AgeTenureInteraction
- ZeroBalance

------------------------------------------------------------

## Machine Learning Models Used

- Logistic Regression
- Decision Tree
- Random Forest
- Gradient Boosting

The best model is selected based on ROC-AUC score.

------------------------------------------------------------

## Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

------------------------------------------------------------

## Explainability

The project includes explainable AI techniques such as:

- Feature Importance Analysis
- Churn Driver Identification

------------------------------------------------------------

## Streamlit Dashboard Features

The dashboard includes:

- Customer Churn Risk Calculator
- Churn Probability Visualization
- Feature Importance Dashboard
- What-if Scenario Simulator
- Model Performance Comparison

------------------------------------------------------------

## Risk Scoring System

The model predicts churn probability and categorizes customers into risk groups:

0.00 – 0.30 → Low Risk
0.30 – 0.70 → Medium Risk
0.70 – 1.00 → High Risk

------------------------------------------------------------

## Project Structure

Bank-Churn-Project/

│

├── data/

├── src/

├── models/

├── reports/

├── streamlit_app/

├── requirements.txt

└── README.md

------------------------------------------------------------

## How to Run the Project

### Install Dependencies

pip install -r requirements.txt

### Run EDA

python src/01_eda.py

### Run Preprocessing

python src/02_preprocessing.py

### Train Models

python src/03_model_training.py

### Generate Explainability Outputs

python src/04_explainability.py

### Run Streamlit Dashboard

streamlit run streamlit_app/app.py

------------------------------------------------------------

## Outputs Generated

The project generates:

- Processed datasets
- Trained machine learning models
- Model comparison reports
- Feature importance reports
- Visualization charts
- Interactive Streamlit dashboard

------------------------------------------------------------

## Business Impact

This project helps banks:

- Detect high-risk customers early
- Improve customer retention
- Optimize engagement strategies
- Reduce revenue loss
- Make data-driven decisions

------------------------------------------------------------

## Future Scope

- SHAP-based customer-level explanations
- Real-time churn monitoring
- XGBoost/LightGBM integration
- Cloud deployment
- Customer segmentation
- Automated retention recommendation system

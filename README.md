# Churn Prediction Model with FastAPI

This project aims to create a machine learning model on the Churn dataset to predict customer churn, whether they will stay or leave. The model takes input data provided via FastAPI and predicts the customer's likelihood of staying or leaving.

## Project Overview

In this project, a machine learning model is developed on the Churn dataset to predict customer churn rate. The model utilizes various customer-related features such as demographic information, account details, and transaction history to determine whether a customer will stay with the company or leave. These predictions serve as valuable insights for companies to improve customer relationships, reduce customer attrition, and make informed strategic decisions.

## Technologies Used

This project leverages the following technologies:

- Python
- FastAPI
- ML libraries and pipeline

## Dataset

The project employs the Churn dataset, which consists of customer-related features. These features include customer identifiers, demographic information, account details, transaction history, and the churn status.


## Installation and Usage

To run this project locally, follow these steps:

- conda create -n fastapi python=3.8

- conda activate fastapi

- pip install -r requirements.txt

- uvicorn main:app --host 0.0.0.0 --port 8002 --reload
  
- click "http://localhost:8002/docs"


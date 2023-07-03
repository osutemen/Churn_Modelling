from fastapi import FastAPI, Request
import joblib
from models import Churn
import numpy as np

# Read models saved during train phase
estimator_loaded = joblib.load( "saved_models/OHE_Churn_model.pkl")

app = FastAPI()

# prediction function
def make_churn_prediction(model, request):
    # parse input from request
    CreditScore = request["CreditScore"]
    Geography =request["Geography"]
    Gender = request["Gender"]
    Age = request["Age"]
    Tenure = request["Tenure"]
    Balance = request["Balance"]
    NumOfProducts = request["NumOfProducts"]
    HasCrCard = request["HasCrCard"]
    IsActiveMember = request["IsActiveMember"]
    EstimatedSalary = request["EstimatedSalary"]




    # Make an input vector
    churn = [[CreditScore , Geography,Gender,Age,Tenure,Balance,NumOfProducts,
            HasCrCard,IsActiveMember,EstimatedSalary ]]
    # Predict,
    prediction = model.predict(churn)
    return prediction

# Churn Prediction endpoint
@app.post("/prediction/churn")
def predict_churn(request: Churn):
    prediction = make_churn_prediction(estimator_loaded, request.dict())
    prediction_label = "stay" if prediction == 0 else "leave"
    return prediction_label



## Churn Prediction endpoint
#@app.post("/prediction/churn")
#def predict_churn(request: Churn):
#    prediction = make_churn_prediction(estimator_loaded, request.dict())
#    print(type(prediction))
#    prediction = prediction.tolist()
#    return prediction[0]
#
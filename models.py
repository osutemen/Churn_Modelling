from pydantic import BaseModel




class Churn(BaseModel):
    CreditScore: float
    Geography: str
    Gender: str
    Age: float
    Tenure: float
    Balance: float
    NumOfProducts: float
    HasCrCard: float
    IsActiveMember: float
    EstimatedSalary: float

    class Config:
        schema_extra = {
            "example": {
                "CreditScore":400,
                "Geography": "France",
                "Gender": "Female",
                "Age": 32,
                "Tenure": 7,
                "Balance": 123.43,
                "NumOfProducts": 3,
                "HasCrCard": 7,
                "IsActiveMember": 1,
                "EstimatedSalary": 76190.75

            }
        }

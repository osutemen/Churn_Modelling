import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score
from sklearn.pipeline import Pipeline

import joblib

# read data
url = "https://raw.githubusercontent.com/erkansirin78/datasets/master/Churn_Modelling.csv"
df = pd.read_csv(url)

print(df.head())
print("---------------------------------------------")
print(df.info())
print("---------------------------------------------")

## One Hot Encoding
#df = pd.get_dummies(df, columns =["Geography", "Gender"], drop_first = True)
#
# Feature matrix
X = df.drop(["Exited","RowNumber", "CustomerId", "Surname"], axis = 1)
X = X.iloc[:,:]
print(X.shape)
print("---------------------------------------------")
print(X.info())
print("---------------------------------------------")

# Output variable
y = df["Exited"]
print(y.shape)

#Create pipeline

pipeline = Pipeline([
    ('ct-ohe', ColumnTransformer(
        [('ct',
          OneHotEncoder(handle_unknown='ignore', categories='auto'),
          [1,2])], remainder='passthrough')
    ),
    ('scaler', StandardScaler(with_mean=False)),
    ('estimator', RandomForestClassifier(n_estimators=200))
])
print(1)
# split test train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Train model with pipeline
pipeline.fit(X_train, y_train)

# Test model
y_pred = pipeline.predict(X_test)

acc = accuracy_score(y_test,y_pred)

print(f"accuracy score:{acc}" )
print(X[:3])
## Save Model
#joblib.dump(pipeline, "saved_models/OHE_Churn_model.pkl")
#
## Make predictions manually
## Read models
#estimator_loaded = joblib.load("saved_models/OHE_Churn_model.pkl")
#
## Prediction set
#X_manual_test = [[400,'France','Male', 32,7,123.43,3,7,1,76190.75]]
#print("X_manual_test", X_manual_test)
#
#prediction = estimator_loaded.predict(X_manual_test)
#print(prediction)
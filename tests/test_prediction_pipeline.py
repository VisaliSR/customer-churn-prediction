import joblib
import pandas as pd
import pytest
customer = pd.DataFrame({
        "gender": ["Female"],
        "SeniorCitizen": [0],
        "Partner": ["Yes"],
        "Dependents": ["No"],
        "tenure": [12],
        "PhoneService": ["Yes"],
        "MultipleLines": ["No"],
        "InternetService": ["Fiber optic"],
        "OnlineSecurity": ["No"],
        "OnlineBackup": ["Yes"],
        "DeviceProtection": ["No"],
        "TechSupport": ["No"],
        "StreamingTV": ["Yes"],
        "StreamingMovies": ["Yes"],
        "Contract": ["Month-to-month"],
        "PaperlessBilling": ["Yes"],
        "PaymentMethod": ["Electronic check"],
        "MonthlyCharges": [85.5],
        "TotalCharges": [1026.0]
    })

def test_prediction_output():    
    pipeline=joblib.load("models/balanced_logistic_regression.pkl")
    prediction=pipeline.predict(customer)
    assert len(prediction)==1
    assert prediction[0] in [0, 1]
def test_prediction_probability():
    pipeline=joblib.load("models/balanced_logistic_regression.pkl")
    prediction=pipeline.predict_proba(customer)
    assert prediction.shape==(1,2)
    assert prediction[0][0]+prediction[0][1]==pytest.approx(1.0)
    
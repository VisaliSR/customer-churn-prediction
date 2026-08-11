import joblib
import pandas as pd


pipeline=joblib.load("models/balanced_logistic_regression.pkl")

new_customer = pd.DataFrame([
    {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0
    }
])
prediction=pipeline.predict(new_customer)
print(f"Customer churn possibility {prediction}")
prob=pipeline.predict_proba(new_customer)
print(f"Customer churn probability {prob}")
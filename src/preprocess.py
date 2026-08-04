import pandas as pd

def clean_data(df):
    #Remove customerID,churn_num
    df=df.drop(columns=["customerID","churn_num"],errors="ignore")

    #Convert TotalCharges into numeric
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

    #fill null values
    df["TotalCharges"]=df["TotalCharges"].fillna(0)

    return df

    # numeric_columns=[
    # "tenure",
    # "MonthlyCharges",
    # "TotalCharges"
    # ]
  
    # one_hot_columns = [
    # "MultipleLines",
    # "InternetService",
    # "OnlineSecurity",
    # "OnlineBackup",
    # "DeviceProtection",
    # "TechSupport",
    # "StreamingTV",
    # "StreamingMovies",
    # "Contract",
    # "PaymentMethod"
    # ]

    # #Binary Encoding
    # mapping={
    #     "Yes":1,"No":0,
    #     "Male":1,"Female":0
    # }
    # for col in binary_columns:
    #     X[col]=X[col].map(mapping)
    # # X["gender"] = X["gender"].map({"Male":0,"Female":1})
    # # X["Partner"] = X["Partner"].map({"Yes":1,"No":0})
    # # X["Depenedents"] = X["Dependents"].map({"Yes":1,"No":0})
    # # X["PhoneService"] = X["PhoneService"].map({"Yes":1,"No":0})
    # # X["PaperlessBilling"] = X["PaperlessBilling"].map({"Yes":1,"No":0})


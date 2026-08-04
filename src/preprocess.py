import pandas as pd

def clean_data(df):
    #Remove customerID,churn_num
    df=df.drop(columns=["customerID","churn_num"],errors="ignore")

    #Convert TotalCharges into numeric
    df["TotalCharges"]=pd.to_numeric(df["TotalCharges"],errors="coerce")

    #fill null values
    df["TotalCharges"]=df["TotalCharges"].fillna(0)

    return df

  
  
   




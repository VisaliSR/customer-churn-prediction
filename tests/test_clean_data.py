from src.preprocess import clean_data
from pandas.api.types import is_numeric_dtype
import pandas as pd

def test_drop_columns():
    df=pd.DataFrame({
        "customerID":["001"],
        "churn_num":1,
        "TotalCharges":["279"]
    })
    result=clean_data(df)
    assert "customerID" not in result.columns
    assert "churn_num" not in  result.columns

def test_totalcharges_numeric():        
    df=pd.DataFrame({
    "TotalCharges":["279.5"]
    })
    result=clean_data(df)
    assert is_numeric_dtype(result["TotalCharges"])

def test_totalcharges_fillna():
      df=pd.DataFrame({
           "TotalCharges":[""]
      })
      result=clean_data(df)
      assert result["TotalCharges"].iloc[0]==0
import pandas as pd
from preprocess import clean_data
from sklearn.model_selection import train_test_split

df=pd.read_csv("../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df=clean_data(df)

y=df["Churn"]
X=df.drop(columns=["Churn"])

y=y.map({"Yes":1,"No":0})

X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.25,
    stratify=y,
    random_state=42
)
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)
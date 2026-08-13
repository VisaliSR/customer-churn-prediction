import pandas as pd
from src.preprocess import clean_data
from src.model import create_preprocessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
df=pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df=clean_data(df)
y=df["Churn"].map({"Yes":1,"No":0})
X=df.drop(columns=["Churn"])
c_values = [0.01, 0.1, 1, 10, 100]
for c in c_values:
    preprocessor=create_preprocessor()
    pipeline=Pipeline(
        [
            ("preprocessor",preprocessor),
            ("model",LogisticRegression(class_weight="balanced",C=c,random_state=42))
        ]
    )
    cv_score=cross_val_score(pipeline,X,y,cv=5,scoring="f1")
    print("Mean :", cv_score.mean())
    print("Standard deviation:", cv_score.std())
import pandas as pd
from sklearn.model_selection import cross_val_score
from src.preprocess import clean_data
from src.model import create_pipeline

df=pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df=clean_data(df)

y=df["Churn"].map({"Yes":1,"No":0})
X=df.drop(columns=["Churn"])
pipeline=create_pipeline()

scores=cross_val_score(pipeline,X,y,cv=5)
print("Scores:", scores)
print("Mean :", scores.mean())
print("Standard deviation:", scores.std())

scores=cross_val_score(pipeline,X,y,cv=5,scoring="recall")
print("Recall scores:", scores)
print("Mean recall:", scores.mean())
print("Recall standard deviation:", scores.std())

scores=cross_val_score(pipeline,X,y,cv=5,scoring="precision")
print("Precision scores:", scores)
print("Mean precision:", scores.mean())
print("Precision standard deviation:", scores.std())

scores = cross_val_score( pipeline, X, y, cv=5, scoring="f1")
print("F1 scores:", scores)
print("Mean F1:", scores.mean())
print("F1 standard deviation:", scores.std())
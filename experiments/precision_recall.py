import pandas as pd
from src.preprocess import clean_data
from src.model import create_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (precision_recall_curve,
    average_precision_score)

df=pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df=clean_data(df)
y=df["Churn"].map({"Yes":1,"No":0})
X=df.drop(columns=["Churn"])
X_train,X_test,y_train,y_test=train_test_split(X,y,stratify=y,random_state=42,test_size=0.25)
pipeline=create_pipeline()
pipeline.fit(X_train,y_train)
churn_probability=pipeline.predict_proba(X_test)[:,1]
precision,recall,thresholds=precision_recall_curve(y_test,churn_probability)
ap = average_precision_score(
    y_test,
    churn_probability
)

print("Average Precision:", ap)
for i in range(5):
    print(
        precision[i],
        recall[i]
    )
print(len(precision),len(recall),len(thresholds))
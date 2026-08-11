import pandas as pd
from src.preprocess import clean_data
from sklearn.model_selection import train_test_split
from src.model import create_pipeline
from sklearn.metrics import f1_score,recall_score,precision_score

threshold=[0.3, 0.4, 0.5, 0.6, 0.7]
df=pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df=clean_data(df)
y=df["Churn"].map({"Yes":1,"No":0})
X=df.drop(columns=["Churn"])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=42,stratify=y)
pipeline=create_pipeline()
pipeline.fit(X_train,y_train)
churn_probability=pipeline.predict_proba(X_test)[:,1]
for i in threshold:
    y_pred=(churn_probability>=i).astype(int)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    print(f"Threshold: {i}" )
    print(f"Precision: {precision:.3f}")
    print(f"Recall: {recall:.3f}")
    print(f"F1: {f1:.3f}")
    print()
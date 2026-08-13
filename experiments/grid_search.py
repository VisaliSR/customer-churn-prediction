import pandas as pd
from sklearn.model_selection import GridSearchCV
from src.preprocess import clean_data
from src.model import create_pipeline
import joblib
df=pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df=clean_data(df)
y=df["Churn"].map({"Yes":1,"No":0})
X=df.drop(columns=["Churn"])
pipeline=create_pipeline()
param_grid={
    "model__C":[0.01, 0.1, 1, 10, 100]
}
grid=GridSearchCV(
    estimator=pipeline,param_grid=param_grid,cv=5,scoring="f1",n_jobs=-1
)
grid.fit(X,y)
print(
    "Best C:",
    grid.best_params_
)

print(
    "Best F1:",
    grid.best_score_
)
joblib.dump(grid.best_estimator_,"models/best_model.pkl")
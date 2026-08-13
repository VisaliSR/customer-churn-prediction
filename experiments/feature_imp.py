import joblib
import pandas as pd


pipeline=joblib.load("models/balanced_logistic_regression.pkl")
preprocessor=pipeline.named_steps["preprocessor"]
model=pipeline.named_steps["model"]

features=preprocessor.get_feature_names_out()
coefficients=model.coef_[0]
imp_df=pd.DataFrame({
    "features":features,
    "coefficients":coefficients
})
print("Top churn inducing features")
print(imp_df.sort_values(by="coefficients",ascending=False).head(10))
print("Least churn reducing features")
print(imp_df.sort_values(by="coefficients",ascending=True).head(10))
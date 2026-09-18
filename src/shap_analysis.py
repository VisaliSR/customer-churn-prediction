import shap
import joblib
from src.preprocess import clean_data
import pandas as pd
import numpy as np

pipeline=joblib.load("models/best_model.pkl")
df=pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")
df=clean_data(df)
preprocessor=pipeline.named_steps["preprocessor"]
model=pipeline.named_steps["model"]
X=df.drop(columns=["Churn"])
X_transformed=preprocessor.transform(X)

explainer=shap.LinearExplainer(model,X_transformed)
customer = X_transformed[0:1]

shap_values=explainer.shap_values(customer)
feature_names = preprocessor.get_feature_names_out()
readable_names = {
    "numeric__tenure": "Tenure",
    "numeric__MonthlyCharges": "Monthly charges",
    "numeric__TotalCharges": "Total charges",
    "binary__gender": "Gender",
    "binary__Partner": "Partner",
    "binary__Dependents": "Dependents",
    "binary__PhoneService": "Phone service",
    "binary__PaperlessBilling": "Paperless billing",
    "remainder__SeniorCitizen": "Senior citizen"
}
shap_df=pd.DataFrame({
    "feature":feature_names,
    "shap_value":shap_values[0]
})
shap_df["abs_shap"] = shap_df["shap_value"].abs()
shap_df = shap_df.sort_values( by="abs_shap",ascending=False)

# Separate factors that increase and decrease churn risk
increasing_risk=shap_df[shap_df["shap_value"]>0]
decreasing_risk=shap_df[shap_df["shap_value"]<0]

# Get top 5 factors in each direction
top_increasing = (
    shap_df[shap_df["shap_value"] > 0]
    .sort_values("abs_shap", ascending=False)
    .head(5)
)

top_decreasing = (
    shap_df[shap_df["shap_value"] < 0]
    .sort_values("abs_shap", ascending=False)
    .head(5)
)

print("Factors increasing churn risk:")
for _, row in top_increasing.iterrows():
    print(f"  {row['feature']} → {row['shap_value']:.3f}")

print("\nFactors decreasing churn risk:")
for _, row in top_decreasing.iterrows():
    print(f"  {row['feature']} → {row['shap_value']:.3f}")
print(feature_names)
# print(shap_df[["feature", "shap_value"]].head(10))

# #for one customer
# shap.plots.waterfall(
#     shap.Explanation(
#         values=shap_values[0],
#         base_values=explainer.expected_value,
#         data=customer[0],
#         feature_names=feature_names
#     )
# )

# #for all the customers to understand the features summary plot
# shap_values_all = explainer.shap_values(X_transformed)
# mean_abs_shap=np.abs(shap_values_all).mean(axis=0)
# importance_df = pd.DataFrame({
#     "feature": feature_names,
#     "mean_abs_shap": mean_abs_shap
# })
# importance_df=importance_df.sort_values(by="mean_abs_shap",ascending=False)
# print(importance_df.head(10))

# shap.summary_plot(
#     shap_values_all,
#     X_transformed,
#     feature_names=feature_names
# )
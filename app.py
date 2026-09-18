import streamlit as st
import joblib
import pandas as pd

def explain_risk(customer):
    reasons = []

    if customer["tenure"].iloc[0] < 12:
        reasons.append("Short customer tenure")

    if customer["Contract"].iloc[0] == "Month-to-month":
        reasons.append("Month-to-month contract")

    if (
        customer["InternetService"].iloc[0] == "Fiber optic"
        and customer["TechSupport"].iloc[0] == "No"
    ):
        reasons.append("Fiber optic service without Tech Support")

    if customer["TechSupport"].iloc[0] == "No":
        reasons.append("No tech support")

    return reasons
st.title("Customer Churn Prediction")
model=joblib.load("models/best_model.pkl")

# st.header("Customer Information")
st.subheader("Personal Information")
col1,col2=st.columns(2)
with col1:
    gender=st.radio("Gender",["Male","Female"])
    partner=st.radio("Partner",["Yes","No"])
    dependents=st.radio("Dependents",["Yes","No"])
with col2:
    senior_citizen=st.checkbox("Senior Citizen")
    tenure=st.number_input(
    "Tenure (months)",
    max_value=100,
    min_value=0,
    value=12
    )


st.subheader("Services")
col1,col2=st.columns(2)
with col1:
    phone_service=st.radio("Phone Service",["Yes","No"])
    multiple_lines=st.selectbox("Multiple Lines",["Yes","No","No phone service"])
    internet_service=st.selectbox(
        "Internet service",
        ["DSL","Fiber optic","No"]
    )
    streaming_tv = st.selectbox(
    "Streaming TV",
    ["Yes", "No", "No internet service"]
    )
    streaming_movies = st.selectbox(
        "Streaming movies",
        ["Yes", "No", "No internet service"]
    )
with col2:
    online_security=st.selectbox(
        "Online Security",
        ["Yes","No","No internet service"]
    )
    online_backup=st.selectbox(
        "Online Backup",
        ["Yes","No","No internet service"]
    )
    tech_support=st.selectbox(
    "Tech Support",
    ["Yes","No","No internet service"]
    )
    device_protection=st.selectbox(
    "Device Protection",
    ["Yes","No","No internet service"]
    )


st.subheader("Billing")
col1, col2 = st.columns(2)
with col1:
    paperless_billing = st.radio(
    "Paperless billing",
    ["Yes", "No"]
    )
    payment_method = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

with col2:
    contract=st.selectbox(
        "Contract type",[
        "Month-to-month",
        "One year",
        "Two year"
        ]
    )
    monthly_charges=st.number_input(
            "Monthly charges",
            min_value=0.0,
            value=85.5
        )
total_charges = tenure * monthly_charges
customer = pd.DataFrame({
    "gender": [gender],
    "SeniorCitizen": [1 if senior_citizen else 0],
    "Partner": [partner],
    "Dependents": [dependents],
    "tenure": [tenure],
    "PhoneService": [phone_service],
    "MultipleLines": [multiple_lines],
    "InternetService": [internet_service],
    "OnlineSecurity": [online_security],
    "OnlineBackup": [online_backup],
    "DeviceProtection": [device_protection],
    "TechSupport": [tech_support],
    "StreamingTV": [streaming_tv],
    "StreamingMovies": [streaming_movies],
    "Contract": [contract],
    "PaperlessBilling": [paperless_billing],
    "PaymentMethod": [payment_method],
    "MonthlyCharges": [monthly_charges],
    "TotalCharges": [total_charges]
})
with st.expander("View Customer Data"):
    st.dataframe(customer)


if st.button("Predict"):
    st.header("Prediction")
    reasons=explain_risk(customer)
    prediction=model.predict(customer)
    probability=float(model.predict_proba(customer)[0][1])

    st.subheader("Customer Summary")
    st.write(f"Tenure: {tenure} months")
    st.write(f"Contract: {contract}")
    st.write(f"Internet Service: {internet_service}")
    st.write(f"Monthly Charges: ${monthly_charges:.2f}")

    st.metric("Churn Probability",f"{probability:.1%}")
    st.progress(probability)
    st.write( "Predicted Class:", "Churn" if prediction[0] == 1 else "No Churn")
    if probability >= 0.7:
        st.error(
            f"High Churn Risk"
        )
        st.write(
        "Consider contacting this customer with a retention offer."
    )

    elif probability >= 0.4:
        st.warning(
            f"Medium Churn Risk"
        )
        st.write(
        "This customer may benefit from proactive engagement."
    )

    else:
        st.success(
            f"Low Churn Risk"
        )
        st.write(
        "This customer currently appears relatively stable."
    )
    st.subheader("Main Risk Factors")
    for reason in reasons:
        st.write("*",reason)
st.caption(
    "Model: Logistic Regression | Dataset: Telco Customer Churn"
)
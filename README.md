# Customer Churn Prediction

An end-to-end machine learning application that predicts customer churn probability and identifies key risk factors for individual customers.

 **[Live Demo](https://customer-churn-prediction-owvg.onrender.com)**

> **Note:** The application is hosted on Render and may take some time to start if it has been inactive.

---

##  Project Overview

Customer churn is an important business problem for subscription-based companies. Identifying customers who are more likely to leave can help businesses take proactive retention measures.

This project uses the **Telco Customer Churn dataset** to build an end-to-end machine learning solution, covering the complete workflow from exploratory data analysis and preprocessing to model training, explainability, web application development, Dockerization, and cloud deployment.

---

## Features

- Predicts customer churn probability
- Classifies customers into **Low, Medium, and High Risk**
- Displays key risk factors for individual predictions
- Interactive Streamlit web interface
- Scikit-learn preprocessing pipeline
- SHAP-based model analysis during development
- Dockerized application
- Cloud deployment using Render

---

##  Dataset

The project uses the **Telco Customer Churn dataset**, containing:

- **7,043 customer records**
- **20 input features**
- **Churn** as the target variable

The dataset contains information about customer demographics, services, contracts, billing, and tenure.

---

##  Exploratory Data Analysis

Exploratory data analysis was performed to understand customer behavior and identify patterns associated with churn.

Some important patterns identified during the analysis include:

- Month-to-month customers had substantially higher churn than customers with one-year or two-year contracts.
- Customers with shorter tenure showed higher churn rates.
- Customers with higher monthly charges showed higher churn in this dataset.
- Fiber optic customers without Tech Support showed a notably higher churn rate.

These observations describe patterns in the dataset and should not be interpreted as causal relationships.

---

##  Data Preprocessing

The preprocessing pipeline includes:

- Handling missing values
- Converting `TotalCharges` to numeric format
- Removing the customer ID
- Binary feature transformation
- One-hot encoding of categorical features
- Feature transformation using `ColumnTransformer`
- Combining preprocessing and model training using a Scikit-learn `Pipeline`

Using a single pipeline ensures that the same preprocessing steps are applied consistently during training and prediction.

---

##  Machine Learning Model

The final model uses **Logistic Regression** for binary classification.

The model predicts the probability that a customer will churn.

The application then uses the predicted probability to categorize customers into three risk levels:

| Churn Probability | Risk Level |
|---|---|
| < 40% | Low Risk |
| 40% – 69% | Medium Risk |
| ≥ 70% | High Risk |

---

##  Model Explainability

SHAP was used during development to understand how different features contributed to model predictions.

The analysis helped identify important features such as:

- Tenure
- Total Charges
- Monthly Charges
- Contract type
- Internet service
- Paperless billing

The deployed application uses a simpler customer-facing explanation based on selected risk factors rather than running SHAP for every prediction.

---

##  Web Application

The prediction interface was built using **Streamlit**.

Users can enter information including:

- Personal information
- Tenure
- Phone and internet services
- Online security and support services
- Contract type
- Payment method
- Monthly charges

The application returns:

- Churn probability
- Predicted churn class
- Risk level
- Main risk factors

##  Docker

The application is containerized using **Docker** to create a reproducible environment for running the application.

### Build the Docker image

```bash
docker build -t customer-churn-app .
```
Run the container
```bash
docker run -p 10000:10000 customer-churn-app
```
The application can then be accessed at:

http://localhost:10000
## Tech Stack

| Technology |	Purpose |
|---|---|
| Python | Programming language |
| Pandas | Data manipulation |
| NumPy	| Numerical operations |
| Scikit-learn	| Machine learning and preprocessing |
| SHAP | Model explainability |
| Streamlit	| Web application |
| Docker | Containerization |
| Render	| Cloud deployment |
| Git & GitHub	| Version control |

## Project Structure

```text
customer-churn-prediction/
│
├── app.py
├── Dockerfile
├── requirements.txt
├── README.md
│
├── models/
│   └── best_model.pkl
│
└── src/
    ├── preprocess.py
    └── shap_analysis.py
```
##Deployment

The Dockerized application is deployed using Render.

The deployment workflow is:

GitHub
   ↓
Dockerfile
   ↓
Docker Image
   ↓
Render
   ↓
Live Streamlit Application

## Future Improvements

Possible future improvements include:

- Model performance monitoring
- Automated model retraining
- Automated testing
- CI/CD pipeline
- More advanced customer-level explanations
- Experimentation with additional classification models

##Author
Visali Rajkumar

An end-to-end machine learning project built to learn and demonstrate the complete workflow from data analysis to deployment.

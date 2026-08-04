import pandas as pd
from preprocess import clean_data
from transformers import BinaryMapper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
# from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix,classification_report



df=pd.read_csv("data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv")

df=clean_data(df)

#Dividing traing and testing data
y=df["Churn"]
X=df.drop(columns=["Churn"])

#Mapping target variable
y=y.map({"Yes":1,"No":0})

#Splitting train and test data 
X_train,X_test,y_train,y_test=train_test_split(
    X,y,
    test_size=0.25,
    stratify=y,
    random_state=42
)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)



#Dividing columns
numeric_columns = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "SeniorCitizen"
]

binary_columns = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]

one_hot_columns = [
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaymentMethod"
]


#Mappings for binary mapping
mappings = {
    "gender": {
        "Male": 1,
        "Female": 0
    },
    "Partner": {
        "Yes": 1,
        "No": 0
    },
    "Dependents": {
        "Yes": 1,
        "No": 0
    },
    "PhoneService": {
        "Yes": 1,
        "No": 0
    },
    "PaperlessBilling": {
        "Yes": 1,
        "No": 0
    }
}
preprocessor= ColumnTransformer(
    transformers=[
        ("numeric",
         StandardScaler(),
         numeric_columns
         ),
         (
            "binary",
            BinaryMapper(mappings),
            binary_columns
         ),
         (
             "categorical",
             OneHotEncoder(handle_unknown="ignore"),
             one_hot_columns
         )
    ],remainder='passthrough'  # Keep remaining columns untouched
)

pipeline=Pipeline(
    [
        ("preprocessor",preprocessor),
        ("model",DecisionTreeClassifier(max_depth=5,random_state=42))
    ]
)
pipeline.fit(X_train,y_train)
y_pred=pipeline.predict(X_test)

matrix=confusion_matrix(y_test,y_pred)
print(matrix)
print(classification_report(y_test,y_pred))
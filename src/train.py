import pandas as pd
import joblib
from preprocess import clean_data
from config import (
    TEST_SIZE,
    RANDOM_STATE,
    NUMERIC_COLUMNS,
    BINARY_COLUMNS,
    ONE_HOT_COLUMNS,
    MAPPINGS
)
from custom_transformers import BinaryMapper
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
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
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RANDOM_STATE
)
# print(X_train.shape)
# print(X_test.shape)
# print(y_train.shape)
# print(y_test.shape)



     
preprocessor= ColumnTransformer(
    transformers=[
        ("numeric",
         StandardScaler(),
         NUMERIC_COLUMNS
         ),
         (
            "binary",
            BinaryMapper(MAPPINGS),
            BINARY_COLUMNS
         ),
         (
             "categorical",
             OneHotEncoder(handle_unknown="ignore"),
             ONE_HOT_COLUMNS
         )
    ],remainder='passthrough'  # Keep remaining columns untouched
)

pipeline=Pipeline(
    [
        ("preprocessor",preprocessor),
        ("model",LogisticRegression(class_weight="balanced",random_state=RANDOM_STATE))
    ]
)
pipeline.fit(X_train,y_train)

joblib.dump(pipeline, "models/balanced_logistic_regression.pkl")
y_pred=pipeline.predict(X_test)


matrix=confusion_matrix(y_test,y_pred)
print(matrix)
print(classification_report(y_test,y_pred))
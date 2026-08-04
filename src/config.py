TEST_SIZE = 0.25
RANDOM_STATE = 42

NUMERIC_COLUMNS = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
    ]
BINARY_COLUMNS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "PaperlessBilling"
]
ONE_HOT_COLUMNS = [
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

MAPPINGS = {
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
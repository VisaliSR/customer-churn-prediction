from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

from src.config import (
    RANDOM_STATE,
    NUMERIC_COLUMNS,
    BINARY_COLUMNS,
    ONE_HOT_COLUMNS,
    MAPPINGS
)

from src.custom_transformers import BinaryMapper


def create_pipeline():

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
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
        ],
        remainder="passthrough"
    )

    pipeline = Pipeline(
        [
            (
                "preprocessor",
                preprocessor
            ),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    random_state=RANDOM_STATE
                )
            )
        ]
    )

    return pipeline
def create_preprocessor():
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
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
        ],
        remainder="passthrough"
    )
    return preprocessor
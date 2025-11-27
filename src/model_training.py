from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

CATEGORICAL_COLS = ["Sex", "Embarked"]
NUMBERICAL_COLS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]

def build_preprocess():
    return ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMBERICAL_COLS),
            ("cat", OneHotEncoder(drop="first"), CATEGORICAL_COLS)
        ]
    )

def build_log_reg_pipeline():
    preprocess = build_preprocess()
    model = Pipeline(steps=[
        ('preprocess', preprocess),
        ('classifier', LogisticRegression(max_iter=1000))
    ])
    return model
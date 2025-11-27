import pandas as pd
from sklearn.model_selection import train_test_split

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]

def load_data(path:str):
    df = pd.read_csv(path)
    return df

def prepare_features(df:pd.DataFrame):
    X = df[FEATURES]
    y = df["Survived"]
    
    X["Age"].fillna(X["Age"].median(), inplace=True)
    X["Embarked"].fillna(X["Embarked"].mode()[0], inplace=True)
    
    return X, y

def train_test_split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
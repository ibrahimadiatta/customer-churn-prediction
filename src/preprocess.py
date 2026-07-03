"""
==========================================================
Customer Churn Prediction
Preprocessing Module

Author : Ibrahima Mbemba Diatta
==========================================================
"""

import os
import joblib
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------
# Load Dataset
# ---------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load dataset.

    Parameters
    ----------
    filepath : str

    Returns
    -------
    DataFrame
    """

    df = pd.read_csv(filepath)

    return df


# ---------------------------------------------------------
# Clean Dataset
# ---------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    data = df.copy()

    # Remove customer ID
    if "customerID" in data.columns:
        data.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges
    data["TotalCharges"] = pd.to_numeric(
        data["TotalCharges"],
        errors="coerce"
    )

    # Missing values
    data["TotalCharges"] = data["TotalCharges"].fillna(
        data["TotalCharges"].median()
    )

    return data


# ---------------------------------------------------------
# Feature Engineering
# ---------------------------------------------------------

def feature_engineering(df: pd.DataFrame):

    data = df.copy()

    data["ChargesPerMonth"] = (
        data["TotalCharges"] /
        (data["tenure"] + 1)
    )

    data["IsNewCustomer"] = (
        data["tenure"] <= 12
    ).astype(int)

    data["HighMonthlyCharges"] = (
        data["MonthlyCharges"] >
        data["MonthlyCharges"].median()
    ).astype(int)

    return data


# ---------------------------------------------------------
# Encode Target
# ---------------------------------------------------------

def encode_target(df):

    data = df.copy()

    data["Churn"] = data["Churn"].map({
        "No": 0,
        "Yes": 1
    })

    return data


# ---------------------------------------------------------
# Encode Features
# ---------------------------------------------------------

def encode_features(df):

    data = df.copy()

    X = data.drop("Churn", axis=1)

    y = data["Churn"]

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    return X, y


# ---------------------------------------------------------
# Scaling
# ---------------------------------------------------------

def scale_features(X):

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler


# ---------------------------------------------------------
# Train Test Split
# ---------------------------------------------------------

def split_data(X, y):

    return train_test_split(

        X,

        y,

        test_size=0.20,

        random_state=42,

        stratify=y

    )


# ---------------------------------------------------------
# Save Objects
# ---------------------------------------------------------

def save_objects(scaler, columns):

    os.makedirs("../models", exist_ok=True)

    joblib.dump(
        scaler,
        "../models/scaler.pkl"
    )

    joblib.dump(
        columns,
        "../models/features.pkl"
    )


# ---------------------------------------------------------
# Save Processed Dataset
# ---------------------------------------------------------

def save_processed(df):

    os.makedirs("../data/processed", exist_ok=True)

    df.to_csv(

        "../data/processed/processed_telco.csv",

        index=False

    )


# ---------------------------------------------------------
# Complete Pipeline
# ---------------------------------------------------------

def preprocess(filepath):

    print("=" * 60)

    print("Loading Dataset...")

    df = load_data(filepath)

    print(df.shape)

    print("=" * 60)

    print("Cleaning Dataset...")

    df = clean_data(df)

    print("=" * 60)

    print("Feature Engineering...")

    df = feature_engineering(df)

    print("=" * 60)

    print("Encoding Target...")

    df = encode_target(df)

    print("=" * 60)

    print("Encoding Features...")

    X, y = encode_features(df)

    print("=" * 60)

    print("Scaling Features...")

    X_scaled, scaler = scale_features(X)

    save_objects(

        scaler,

        X.columns.tolist()

    )

    save_processed(df)

    X_train, X_test, y_train, y_test = split_data(
        X_scaled,
        y
    )

    print("=" * 60)
    print("Applying SMOTE...")

    smote = SMOTE(random_state=42)

    X_train, y_train = smote.fit_resample(
        X_train,
        y_train
    )

    print("=" * 60)

    print("Preprocessing Completed Successfully!")

    return (

        X_train,

        X_test,

        y_train,

        y_test,

        scaler,

        X.columns

    )

smote = SMOTE(
    random_state=42
)



# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    preprocess(

        "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"

    )

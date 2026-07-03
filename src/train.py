"""
==========================================================
Customer Churn Prediction
Model Training Module

Author : Ibrahima Mbemba Diatta
==========================================================
"""

import os
import joblib
import pandas as pd

from preprocess import preprocess

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import GridSearchCV

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

# ---------------------------------------------------------
# Optional XGBoost
# ---------------------------------------------------------

try:
    from xgboost import XGBClassifier
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ---------------------------------------------------------
# Build Models
# ---------------------------------------------------------

def build_models():

    models = {

        "Logistic Regression":
            LogisticRegression(
                random_state=42,
                max_iter=1000
            ),

        "KNN":
            KNeighborsClassifier(),

        "Decision Tree":
            DecisionTreeClassifier(
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                random_state=42
            ),

        "Gradient Boosting":
            GradientBoostingClassifier(
                random_state=42
            )

    }

    if XGB_AVAILABLE:

        models["XGBoost"] = XGBClassifier(
            random_state=42,
            eval_metric="logloss"
        )

    return models


# ---------------------------------------------------------
# Evaluate Model
# ---------------------------------------------------------

def evaluate(model, X_test, y_test):

    predictions = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X_test)[:,1]
    else:
        probabilities = predictions

    return {

        "Accuracy":
            accuracy_score(
                y_test,
                predictions
            ),

        "Precision":
            precision_score(
                y_test,
                predictions
            ),

        "Recall":
            recall_score(
                y_test,
                predictions
            ),

        "F1":
            f1_score(
                y_test,
                predictions
            ),

        "ROC-AUC":
            roc_auc_score(
                y_test,
                probabilities
            )

    }


# ---------------------------------------------------------
# Hyperparameter Tuning
# ---------------------------------------------------------

def optimize_random_forest(X_train, y_train):

    parameters = {

        "n_estimators":[100,200,300],

        "max_depth":[
            None,
            5,
            10,
            15
        ],

        "min_samples_split":[
            2,
            5,
            10
        ]

    }

    grid = GridSearchCV(

        RandomForestClassifier(random_state=42),

        param_grid=parameters,

        cv=5,

        scoring="f1",

        n_jobs=-1

    )

    grid.fit(X_train, y_train)

    print("\nBest Random Forest Parameters")

    print(grid.best_params_)

    return grid.best_estimator_


# ---------------------------------------------------------
# Training Pipeline
# ---------------------------------------------------------

def train():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        scaler,
        columns
    ) = preprocess(
        "../data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
    )

    models = build_models()

    results = []

    trained_models = {}

    print("="*60)

    print("Training Models...")

    print("="*60)

    for name, model in models.items():

        print(f"\n{name}")

        model.fit(
            X_train,
            y_train
        )

        trained_models[name] = model

        metrics = evaluate(
            model,
            X_test,
            y_test
        )

        metrics["Model"] = name

        results.append(metrics)

        print(metrics)

    print("\nOptimizing Random Forest...")

    best_rf = optimize_random_forest(
        X_train,
        y_train
    )

    trained_models["Optimized Random Forest"] = best_rf

    metrics = evaluate(
        best_rf,
        X_test,
        y_test
    )

    metrics["Model"] = "Optimized Random Forest"

    results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1",
            "ROC-AUC"
        ]
    ]

    results_df.sort_values(

        by="ROC-AUC",

        ascending=False,

        inplace=True

    )

    results_df.reset_index(
        drop=True,
        inplace=True
    )

    print("\n")
    print("="*60)
    print("Model Comparison")
    print("="*60)

    print(results_df)

    best_model_name = results_df.iloc[0]["Model"]

    best_model = trained_models[best_model_name]

    os.makedirs("../models", exist_ok=True)

    joblib.dump(

        best_model,

        "../models/best_model.pkl"

    )

    results_df.to_csv(

        "../models/model_results.csv",

        index=False

    )

    print("\n")
    print("="*60)

    print("Training Finished")

    print(f"Best Model : {best_model_name}")

    print("Saved : models/best_model.pkl")

    print("="*60)

    return (

        best_model,

        results_df

    )


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------

if __name__ == "__main__":

    train()

"""
==========================================================
Customer Churn Prediction
Model Evaluation Module

Author : Ibrahima Mbemba Diatta
==========================================================
"""

import os
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    precision_recall_curve
)

from preprocess import preprocess


# ---------------------------------------------------------
# Load Trained Model
# ---------------------------------------------------------

def load_model():

    return joblib.load("../models/best_model.pkl")


# ---------------------------------------------------------
# Predictions
# ---------------------------------------------------------

def predict(model, X_test):

    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):

        y_prob = model.predict_proba(X_test)[:,1]

    else:

        y_prob = y_pred

    return y_pred, y_prob


# ---------------------------------------------------------
# Evaluation Metrics
# ---------------------------------------------------------

def evaluate_metrics(y_test, y_pred, y_prob):

    print("="*60)
    print("MODEL PERFORMANCE")
    print("="*60)

    print(f"Accuracy  : {accuracy_score(y_test,y_pred):.4f}")
    print(f"Precision : {precision_score(y_test,y_pred):.4f}")
    print(f"Recall    : {recall_score(y_test,y_pred):.4f}")
    print(f"F1 Score  : {f1_score(y_test,y_pred):.4f}")
    print(f"ROC AUC   : {roc_auc_score(y_test,y_prob):.4f}")

    print("\nClassification Report\n")

    print(classification_report(y_test,y_pred))


# ---------------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------------

def plot_confusion_matrix(y_test,y_pred):

    os.makedirs("../images",exist_ok=True)

    cm = confusion_matrix(y_test,y_pred)

    plt.figure(figsize=(6,5))

    sns.heatmap(

        cm,

        annot=True,

        fmt="d",

        cmap="Blues",

        xticklabels=["No","Yes"],

        yticklabels=["No","Yes"]

    )

    plt.title("Confusion Matrix")

    plt.xlabel("Predicted")

    plt.ylabel("Actual")

    plt.tight_layout()

    plt.savefig("../images/confusion_matrix.png")

    plt.show()


# ---------------------------------------------------------
# ROC Curve
# ---------------------------------------------------------

def plot_roc(y_test,y_prob):

    fpr,tpr,_ = roc_curve(y_test,y_prob)

    plt.figure(figsize=(7,6))

    plt.plot(

        fpr,

        tpr,

        linewidth=2,

        label="ROC Curve"

    )

    plt.plot([0,1],[0,1],"--")

    plt.xlabel("False Positive Rate")

    plt.ylabel("True Positive Rate")

    plt.title("ROC Curve")

    plt.legend()

    plt.tight_layout()

    plt.savefig("../images/roc_curve.png")

    plt.show()


# ---------------------------------------------------------
# Precision Recall Curve
# ---------------------------------------------------------

def plot_pr_curve(y_test,y_prob):

    precision,recall,_ = precision_recall_curve(

        y_test,

        y_prob

    )

    plt.figure(figsize=(7,6))

    plt.plot(

        recall,

        precision,

        linewidth=2

    )

    plt.xlabel("Recall")

    plt.ylabel("Precision")

    plt.title("Precision Recall Curve")

    plt.tight_layout()

    plt.savefig("../images/precision_recall_curve.png")

    plt.show()


# ---------------------------------------------------------
# Feature Importance
# ---------------------------------------------------------

def feature_importance(model,columns):

    if not hasattr(model,"feature_importances_"):

        print("Feature Importance not available.")

        return

    importance = pd.DataFrame({

        "Feature":columns,

        "Importance":model.feature_importances_

    })

    importance = importance.sort_values(

        by="Importance",

        ascending=False

    )

    plt.figure(figsize=(10,8))

    sns.barplot(

        data=importance.head(15),

        x="Importance",

        y="Feature"

    )

    plt.title("Top 15 Important Features")

    plt.tight_layout()

    plt.savefig("../images/feature_importance.png")

    plt.show()


# ---------------------------------------------------------
# Main Evaluation Pipeline
# ---------------------------------------------------------

def main():

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

    model = load_model()

    y_pred,y_prob = predict(

        model,

        X_test

    )

    evaluate_metrics(

        y_test,

        y_pred,

        y_prob

    )

    plot_confusion_matrix(

        y_test,

        y_pred

    )

    plot_roc(

        y_test,

        y_prob

    )

    plot_pr_curve(

        y_test,

        y_prob

    )

    feature_importance(

        model,

        columns

    )

    print("\nEvaluation completed successfully!")


if __name__ == "__main__":

    main()

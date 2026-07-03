"""
==========================================================
Customer Churn Prediction
Utility Functions

Author : Ibrahima Mbemba Diatta
==========================================================
"""

import os
import joblib
import logging
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# ==========================================================
# Logger Configuration
# ==========================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ==========================================================
# Create Directory
# ==========================================================

def create_directory(path: str):

    """
    Create directory if it does not exist.
    """

    os.makedirs(path, exist_ok=True)


# ==========================================================
# Save Object
# ==========================================================

def save_object(object_, filepath):

    """
    Save Python object.
    """

    create_directory(os.path.dirname(filepath))

    joblib.dump(object_, filepath)

    logger.info(f"Saved : {filepath}")


# ==========================================================
# Load Object
# ==========================================================

def load_object(filepath):

    """
    Load Python object.
    """

    logger.info(f"Loading : {filepath}")

    return joblib.load(filepath)


# ==========================================================
# Metrics
# ==========================================================

def compute_metrics(y_true, y_pred, y_prob):

    """
    Compute evaluation metrics.
    """

    return {

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred
        ),

        "Recall": recall_score(
            y_true,
            y_pred
        ),

        "F1 Score": f1_score(
            y_true,
            y_pred
        ),

        "ROC AUC": roc_auc_score(
            y_true,
            y_prob
        )

    }


# ==========================================================
# Display Metrics
# ==========================================================

def print_metrics(metrics):

    """
    Display metrics.
    """

    print("=" * 60)

    print("MODEL PERFORMANCE")

    print("=" * 60)

    for metric, value in metrics.items():

        print(f"{metric:<15}: {value:.4f}")

    print("=" * 60)


# ==========================================================
# Save Results
# ==========================================================

def save_results(results, filepath):

    """
    Save model comparison results.
    """

    create_directory(os.path.dirname(filepath))

    if isinstance(results, pd.DataFrame):

        results.to_csv(filepath, index=False)

    logger.info(f"Results saved : {filepath}")


# ==========================================================
# Project Banner
# ==========================================================

def banner():

    print("=" * 60)

    print("Customer Churn Prediction")

    print("Machine Learning Pipeline")

    print("Author : Ibrahima Mbemba Diatta")

    print("=" * 60)

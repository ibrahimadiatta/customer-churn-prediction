"""
==========================================================
Customer Churn Prediction
Prediction Module

Author : Ibrahima Mbemba Diatta
==========================================================
"""

import joblib
import pandas as pd


# ---------------------------------------------------------
# Load Saved Objects
# ---------------------------------------------------------

def load_model():

    model = joblib.load("../models/best_model.pkl")

    scaler = joblib.load("../models/scaler.pkl")

    features = joblib.load("../models/features.pkl")

    return model, scaler, features


# ---------------------------------------------------------
# Prepare Input Data
# ---------------------------------------------------------

def prepare_input(customer_data, features):

    customer = pd.DataFrame([customer_data])

    customer = pd.get_dummies(customer)

    customer = customer.reindex(

        columns=features,

        fill_value=0

    )

    return customer


# ---------------------------------------------------------
# Predict Customer Churn
# ---------------------------------------------------------

def predict(customer_data):

    model, scaler, features = load_model()

    customer = prepare_input(

        customer_data,

        features

    )

    customer_scaled = scaler.transform(customer)

    prediction = model.predict(customer_scaled)[0]

    probability = model.predict_proba(

        customer_scaled

    )[0]

    return prediction, probability


# ---------------------------------------------------------
# Display Prediction
# ---------------------------------------------------------

def display_result(prediction, probability):

    print("=" * 60)

    print("CUSTOMER CHURN PREDICTION")

    print("=" * 60)

    if prediction == 1:

        print("Prediction : Customer WILL churn.")

    else:

        print("Prediction : Customer will NOT churn.")

    print()

    print(f"Probability of Staying : {probability[0]:.2%}")

    print(f"Probability of Churning : {probability[1]:.2%}")

    print("=" * 60)


# ---------------------------------------------------------
# Example Customer
# ---------------------------------------------------------

if __name__ == "__main__":

    new_customer = {

        "gender": "Female",

        "SeniorCitizen": 0,

        "Partner": "Yes",

        "Dependents": "No",

        "tenure": 10,

        "PhoneService": "Yes",

        "MultipleLines": "No",

        "InternetService": "Fiber optic",

        "OnlineSecurity": "No",

        "OnlineBackup": "Yes",

        "DeviceProtection": "No",

        "TechSupport": "No",

        "StreamingTV": "Yes",

        "StreamingMovies": "Yes",

        "Contract": "Month-to-month",

        "PaperlessBilling": "Yes",

        "PaymentMethod": "Electronic check",

        "MonthlyCharges": 89.50,

        "TotalCharges": 895.00,

        "ChargesPerMonth": 81.36,

        "IsNewCustomer": 1,

        "HighMonthlyCharges": 1

    }

    prediction, probability = predict(new_customer)

    display_result(

        prediction,

        probability

    )

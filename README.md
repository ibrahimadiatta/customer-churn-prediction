<div align="center">

# 📉 Customer Churn Prediction using Machine Learning

### End-to-End Machine Learning Project following the CRISP-DM Methodology

**Author:** Ibrahima Mbemba Diatta

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Latest-orange.svg)](https://scikit-learn.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.x-blue.svg)](https://pandas.pydata.org/)
[![NumPy](https://img.shields.io/badge/NumPy-Latest-success.svg)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-Latest-red.svg)](https://matplotlib.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

</div>

---

# 📌 Project Overview

Customer churn is one of the biggest challenges faced by telecommunication companies.

Losing existing customers directly impacts company revenue and increases customer acquisition costs.

The objective of this project is to build an end-to-end Machine Learning pipeline capable of predicting whether a customer is likely to leave the company.

This project follows the **CRISP-DM methodology**, the industry standard process for Data Science projects.

---

# 🎯 Business Objectives

The project aims to:

- Predict customer churn.
- Identify the most influential factors.
- Compare multiple Machine Learning algorithms.
- Select the best predictive model.
- Build a reusable Machine Learning pipeline.

---

# 📊 Dataset

**Dataset:** IBM Telco Customer Churn

The dataset contains customer information including:

- Customer demographics
- Services subscribed
- Billing information
- Customer tenure
- Monthly charges
- Total charges
- Churn status

Target Variable

```
Churn
```

---

# 🧠 Machine Learning Workflow

The project follows the CRISP-DM methodology.

```
Business Understanding
        ↓
Data Understanding
        ↓
Data Preparation
        ↓
Modeling
        ↓
Evaluation
        ↓
Deployment
```

---

# 📁 Project Structure

```text
customer-churn-prediction/

│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── Customer_Churn_Prediction_CRISP_DM.ipynb
│
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── models/
│   ├── best_model.pkl
│   ├── scaler.pkl
│   ├── features.pkl
│   └── model_results.csv
│
├── images/
│
├── README.md
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

# 🔎 Exploratory Data Analysis

The exploratory analysis includes:

- Missing values analysis
- Duplicate detection
- Target distribution
- Feature distributions
- Correlation analysis
- Customer behavior visualization

Example visualizations:

- Customer Churn Distribution
- Correlation Heatmap
- Monthly Charges Distribution
- Tenure Distribution
- Feature Importance

---

# ⚙️ Data Preparation

The preprocessing pipeline performs:

- Customer ID removal
- Missing value imputation
- Feature Engineering
- One-Hot Encoding
- Feature Scaling
- Train/Test Split

Feature Engineering includes:

- ChargesPerMonth
- HighMonthlyCharges
- IsNewCustomer

---

# 🤖 Machine Learning Models

Several Machine Learning algorithms were trained and compared.

| Model | Description |
|--------|-------------|
| Logistic Regression | Linear baseline model |
| K-Nearest Neighbors | Distance-based classifier |
| Decision Tree | Tree-based classifier |
| Random Forest | Ensemble learning |
| Gradient Boosting | Boosting algorithm |
| XGBoost | Extreme Gradient Boosting |

---

# 📈 Evaluation Metrics

Models are evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score
- ROC-AUC

Additional evaluation includes:

- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

---

# 🏆 Best Model

The best-performing model is automatically selected based on the evaluation metrics and saved as:

```
models/best_model.pkl
```

---

# 🚀 Installation

Clone the repository

```bash
git clone https://github.com/ibrahimadiatta/customer-churn-prediction.git
```

Move into the project

```bash
cd customer-churn-prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Project

Preprocessing

```bash
python src/preprocess.py
```

Training

```bash
python src/train.py
```

Evaluation

```bash
python src/evaluate.py
```

Prediction

```bash
python src/predict.py
```

---

# 📊 Results

The project automatically generates:

- Model comparison table
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

All figures are saved inside:

```
images/
```

---

# 📚 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- Matplotlib
- Seaborn
- XGBoost
- Joblib

---

# 📌 Future Improvements

Possible extensions include:

- Hyperparameter Optimization using Optuna
- SHAP Explainability
- Feature Selection
- FastAPI REST API
- Docker Containerization
- CI/CD with GitHub Actions
- Cloud Deployment (AWS, Azure)

---

# 👨‍💻 Author

**Ibrahima Mbemba Diatta**

Master's Student in Data Science & Artificial Intelligence

GitHub:

https://github.com/ibrahimadiatta

LinkedIn:

www.linkedin.com/in/ibrahima-mbemba-diatta

---

# ⭐ If you find this project useful

Please consider giving it a ⭐ on GitHub.

It helps support the project and motivates future improvements.

---

# 📄 License

This project is licensed under the MIT License.
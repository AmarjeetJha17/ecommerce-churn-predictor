# E-Commerce Customer Churn Predictor & Retention System

## Architecture Overview
This repository contains an end-to-end Machine Learning pipeline and full-stack web dashboard designed to predict e-commerce customer churn. By identifying high-risk customers before they abandon the platform, the system allows store managers to trigger targeted retention strategies.

## Key Features
* **Robust Preprocessing:** Automated `scikit-learn` Pipeline utilizing `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` to process highly skewed transactional data without leakage.
* **Class Imbalance Handling:** Integrated `SMOTE` via `imbalanced-learn` strictly on the training folds to balance the 16% churn minority class.
* **Advanced Ensembling:** A `VotingClassifier` leveraging Logistic Regression, Random Forest, and a highly tuned LightGBM model optimized via Bayesian optimization (`Optuna`).
* **Business-Tuned Thresholds:** The decision threshold was mathematically lowered from 0.5 to 0.35 to prioritize **Recall**, ensuring zero False Negatives in catching critical flight-risk accounts.
* **Model Interpretability:** Global feature importance extraction utilizing `SHAP` to transform the "black box" ensemble into clear, actionable business drivers (e.g., Complaint History, Tenure).
* **Full-Stack Deployment:** The serialized `.joblib` model is served via a Flask REST API, interacting with a SQLite historical database and a responsive HTML/CSS frontend dashboard.

## Tech Stack
`Python` | `Pandas` | `Scikit-Learn` | `LightGBM` | `Optuna` | `SHAP` | `Flask` | `SQLite`

## System Performance
* **Recall (Class 1):** 1.00 (Zero Missed Churners at custom threshold)
* **ROC-AUC Score:** 0.89+

## How to Run Locally
1. Clone the repository and run `pip install -r requirements.txt`.
2. Navigate to the `api/` directory.
3. Run `python seed_db.py` to initialize the SQLite database with historical records.
4. Run `python app.py` to start the Flask server.
5. Open `http://localhost:5000` to access the Store Manager Dashboard.
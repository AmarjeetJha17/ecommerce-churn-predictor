# 🛒 E-Commerce Customer Churn Predictor & Retention System

![Python](https://img.shields.io/badge/Python-3.13+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-3.1+-lightgrey.svg)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6+-green.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-orange.svg)
![uv](https://img.shields.io/badge/uv-Fast_Python_Packaging-purple.svg)

An end-to-end Machine Learning pipeline and full-stack web dashboard designed to predict e-commerce customer churn. By identifying high-risk customers before they abandon the platform, the system allows store managers to trigger targeted retention strategies.

## 🚀 Key Features

* **Robust Preprocessing:** Automated `scikit-learn` Pipeline utilizing `SimpleImputer`, `StandardScaler`, and `OneHotEncoder` to process highly skewed transactional data without leakage.
* **Class Imbalance Handling:** Integrated `SMOTE` via `imbalanced-learn` strictly on the training folds to balance the 16% churn minority class.
* **Advanced Ensembling:** A `VotingClassifier` leveraging Logistic Regression, Random Forest, and a highly tuned LightGBM model optimized via Bayesian optimization (`Optuna`).
* **Business-Tuned Thresholds:** The decision threshold was mathematically lowered from 0.5 to 0.35 to prioritize **Recall**, ensuring zero False Negatives in catching critical flight-risk accounts.
* **Model Interpretability:** Global feature importance extraction utilizing `SHAP` to transform the "black box" ensemble into clear, actionable business drivers (e.g., Complaint History, Tenure).
* **Full-Stack Deployment:** The serialized `.joblib` model is served via a Flask REST API, interacting with a SQLite historical database and a responsive HTML/CSS frontend dashboard.

## 🧠 System Performance

* **Recall (Class 1):** 1.00 (Zero Missed Churners at custom threshold)
* **ROC-AUC Score:** 0.89+

## 📁 Project Structure

```text
ecommerce-churn-predictor/
├── api/                    # Flask backend and UI
│   ├── static/             # Static assets (CSS, JS, etc.)
│   ├── templates/          # HTML templates for the dashboard
│   ├── app.py              # Main Flask application
│   ├── seed_db.py          # Script to initialize the SQLite database
│   └── add_new_customer.py # Script for adding mock customers
├── artifacts/              # Serialized ML models (e.g., joblib pipelines)
├── data/                   # Raw and processed datasets
│   ├── raw/
│   └── processed/
├── notebooks/              # Jupyter notebooks for model development
│   ├── 01_eda_and_cleaning.ipynb
│   ├── 02_preprocessing_and_baseline.ipynb
│   └── 03_ensembling_and_tuning.ipynb
├── pyproject.toml          # Project configuration and dependencies
├── uv.lock                 # Lockfile for reproducible builds via uv
└── README.md               # Project documentation
```

## 🛠️ Tech Stack

- **Machine Learning & Data Processing:** `Python`, `Pandas`, `Scikit-Learn`, `LightGBM`, `imbalanced-learn`, `Optuna`, `SHAP`
- **Backend & API:** `Flask`, `SQLite`, `SQLAlchemy`
- **Frontend Dashboard:** Vanilla `HTML`, `CSS`, `JavaScript`
- **Package Management:** `uv`

## ⚙️ How to Run Locally

This project uses modern Python packaging with `uv` for lightning-fast dependency management, but standard `pip` can also be used.

### Prerequisites
- Python 3.13 or higher
- [uv](https://github.com/astral-sh/uv) (Optional but recommended)

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd ecommerce-churn-predictor
   ```

2. **Install Dependencies:**
   - **Using `uv` (Recommended):**
     ```bash
     uv sync
     # or
     uv pip install -e .
     ```
   - **Using standard `pip`:**
     ```bash
     pip install -e .
     ```

3. **Initialize the Database:**
   Navigate to the `api` directory and seed the SQLite database with historical records.
   ```bash
   cd api
   python seed_db.py
   ```

4. **Start the Flask Server:**
   ```bash
   python app.py
   ```

5. **Access the Dashboard:**
   Open your browser and navigate to:
   [http://localhost:5000](http://localhost:5000)

## 🖥️ Usage

Once the Store Manager Dashboard is running, you can:
- **Database Lookup:** Enter an existing `Customer ID` to fetch their profile, calculate the current churn probability, and see AI insights. This automatically pre-fills the manual form for "what-if" testing.
- **Manual Simulation:** Manually adjust parameters (like *Tenure*, *Recent Complaints*, or *Cashback Amount*) to simulate and test different retention strategies on the fly. The dashboard leverages `SHAP` to highlight the exact driver influencing the prediction.
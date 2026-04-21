# api/app.py
from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import joblib
import shap
import os

app = Flask(__name__)

pipeline_path = '../artifacts/churn_pipeline_v1.joblib'
try:
    model_pipeline = joblib.load(pipeline_path)
    print("ML Pipeline loaded successfully.")
    
    # Initialize SHAP Explainer
    preprocessor = model_pipeline.named_steps['preprocessor']
    lgbm_model = model_pipeline.named_steps['classifier'].estimators_[2]
    explainer = shap.TreeExplainer(lgbm_model)
    print("SHAP Explainer initialized.")
    
except Exception as e:
    print(f"CRITICAL ERROR: Could not load pipeline: {e}")

DB_PATH = 'churn_system.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.errorhandler(404)
def not_found_error(error):
    return jsonify({'error': 'Resource not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error. Please check the backend logs.'}), 500

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_churn():
    try:
        data = request.get_json()
        customer_id = data.get('customer_id')
        
        if not customer_id:
            return jsonify({'error': 'Customer ID is required'}), 400

        conn = get_db_connection()
        query = "SELECT * FROM customer_data WHERE CustomerID = ?"
        customer_data = pd.read_sql_query(query, conn, params=(customer_id,))
        
        if customer_data.empty:
            conn.close()
            return jsonify({'error': f'Customer {customer_id} not found in database'}), 404
            
        features_df = customer_data.drop(['CustomerID', 'Churn'], axis=1, errors='ignore')
        
        # Run inference
        probability = model_pipeline.predict_proba(features_df)[0][1]
        threshold = 0.35
        is_at_risk = bool(probability >= threshold)
        
        # Transform the single customer's data using the pipeline's preprocessor
        X_transformed = preprocessor.transform(features_df)
        shap_values = explainer.shap_values(X_transformed)
        
        # Handle SHAP output format (class 1 is churn)
        individual_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        
        # Extract feature names that the model memorized, ignoring SQLite data types
        num_cols = preprocessor.transformers_[0][2] 
        cat_cols = preprocessor.transformers_[1][2] 
        
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_names = cat_encoder.get_feature_names_out(cat_cols)
        all_feature_names = list(num_cols) + list(cat_names)
        
        # Map SHAP values to feature names and find the top driver for churn (highest positive value)
        impacts = {name: val for name, val in zip(all_feature_names, individual_shap)}
        top_driver = sorted(impacts.items(), key=lambda item: item[1], reverse=True)[0]
        
        # Clean up the feature name for the frontend (e.g., "Complain" -> "Complain")
        top_risk_factor = top_driver[0].replace('_', ' ')
        
        # Handle missing (NaN/None) values for the dashboard display
        # If Tenure or Complain is missing in the DB, default it to 0 for the UI
        raw_tenure = customer_data['Tenure'].values[0]
        raw_complain = customer_data['Complain'].values[0]
        
        safe_tenure = int(raw_tenure) if pd.notna(raw_tenure) else 0
        safe_complain = int(raw_complain) if pd.notna(raw_complain) else 0

        # Log to database
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO predictions (CustomerID, churn_probability) VALUES (?, ?)",
            (customer_id, float(probability))
        )
        conn.commit()
        conn.close()
        
        return jsonify({
            'customer_id': customer_id,
            'churn_probability': round(probability * 100, 2),
            'is_at_risk': is_at_risk,
            'tenure': safe_tenure,
            'complaints': safe_complain,
            'top_risk_factor': top_risk_factor
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
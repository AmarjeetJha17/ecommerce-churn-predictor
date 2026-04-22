# api/app.py
from flask import Flask, render_template, request, jsonify
import sqlite3
import pandas as pd
import joblib
import shap
import traceback

app = Flask(__name__)

# 1. Load the trained pipeline at server startup
pipeline_path = '../artifacts/churn_pipeline_v1.joblib'
try:
    model_pipeline = joblib.load(pipeline_path)
    print("ML Pipeline loaded successfully.")
    
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

def calculate_shap_and_prediction(features_df):
    """Helper function to run inference and SHAP for both routes"""
    # 1. Predict
    probability = model_pipeline.predict_proba(features_df)[0][1]
    is_at_risk = bool(probability >= 0.35)
    
    # 2. SHAP
    X_transformed = preprocessor.transform(features_df)
    shap_values = explainer.shap_values(X_transformed)
    individual_shap = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
    
    # 3. Extract features
    num_cols = preprocessor.transformers_[0][2] 
    cat_cols = preprocessor.transformers_[1][2] 
    try:
        cat_encoder = preprocessor.named_transformers_['cat'].named_steps['onehot']
        cat_names = cat_encoder.get_feature_names_out(cat_cols)
        all_feature_names = list(num_cols) + list(cat_names)
    except:
        all_feature_names = list(features_df.columns)
        
    impacts = {name: val for name, val in zip(all_feature_names, individual_shap[:len(all_feature_names)])}
    top_driver = sorted(impacts.items(), key=lambda item: item[1], reverse=True)[0]
    top_risk_factor = top_driver[0].replace('_', ' ')
    
    return probability, is_at_risk, top_risk_factor

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# --- ROUTE 1: Database Lookup ---
@app.route('/predict_db', methods=['POST'])
def predict_db():
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
        
        probability, is_at_risk, top_risk_factor = calculate_shap_and_prediction(features_df)
        
        # Log to DB
        cursor = conn.cursor()
        cursor.execute("INSERT INTO predictions (CustomerID, churn_probability) VALUES (?, ?)", (customer_id, float(probability)))
        conn.commit()
        conn.close()
        
        # Convert the customer's raw data to a dictionary so the frontend can pre-fill the manual form
        raw_features = features_df.iloc[0].to_dict()
        
        return jsonify({
            'source': 'database',
            'customer_id': customer_id,
            'churn_probability': round(probability * 100, 2),
            'is_at_risk': is_at_risk,
            'top_risk_factor': top_risk_factor,
            'raw_features': raw_features # Sending raw data back to UI
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

# --- ROUTE 2: Manual Simulation ---
@app.route('/predict_manual', methods=['POST'])
def predict_manual():
    try:
        data = request.get_json()
        features_df = pd.DataFrame([data])
        
        probability, is_at_risk, top_risk_factor = calculate_shap_and_prediction(features_df)
        
        return jsonify({
            'source': 'manual',
            'churn_probability': round(probability * 100, 2),
            'is_at_risk': is_at_risk,
            'top_risk_factor': top_risk_factor
        })
        
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
import pandas as pd
import sqlite3
import os

db_path = 'churn_system.db'
excel_path = '../data/raw/E Commerce Dataset.xlsx'

def seed_database():
    print("Loading historical data...")
    df = pd.read_excel(excel_path, sheet_name='E Comm')
    
    conn = sqlite3.connect(db_path)
    
    df.to_sql('customer_data', conn, if_exists='replace', index=False)
    
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            prediction_id INTEGER PRIMARY KEY AUTOINCREMENT,
            CustomerID TEXT,
            churn_probability REAL,
            prediction_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    print(f"Database successfully seeded at {db_path} with {len(df)} records.")

if __name__ == '__main__':
    seed_database()
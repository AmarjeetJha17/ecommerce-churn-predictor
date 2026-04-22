# api/add_new_customer.py
import sqlite3
import pandas as pd

def inject_custom_record():
    conn = sqlite3.connect('churn_system.db')
    cursor = conn.cursor()
    
    new_customer = {
        'CustomerID': '99999',
        'Tenure': 1.0,                      
        'CityTier': 3,
        'WarehouseToHome': 25.0,
        'HourSpendOnApp': 2.0,
        'NumberOfDeviceRegistered': 4,
        'SatisfactionScore': 1,              
        'NumberOfAddress': 1,
        'Complain': 1,                       
        'OrderAmountHikeFromlastYear': 11.0,
        'CouponUsed': 0.0,
        'OrderCount': 1.0,
        'DaySinceLastOrder': 15.0,
        'CashbackAmount': 110.0,             
        'PreferedOrderCat': 'Mobile Phone',
        'PreferredPaymentMode': 'Cash on Delivery',
        'Gender': 'Male',
        'MaritalStatus': 'Single',
        'PreferredLoginDevice': 'Mobile Phone',
        'Churn': 0                           
    }
    
    # 2. Convert to DataFrame
    df_new = pd.DataFrame([new_customer])
    
    # 3. Append to the customer_data table
    try:
        df_new.to_sql('customer_data', conn, if_exists='append', index=False)
        print("Success! Custom record ID 99999 injected into the database.")
    except Exception as e:
        print(f"Error injecting record: {e}")
        
    conn.close()

if __name__ == '__main__':
    inject_custom_record()
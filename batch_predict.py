import pandas as pd
from sqlalchemy import create_engine
import mlflow.sklearn
from datetime import datetime
import schedule
import time

DB_URL = 'sqlite:///local_database.db'
engine = create_engine(DB_URL)

def load_model():
    try:
        model_uri = "models:/GermanCredit_LGBM/Latest"
        model = mlflow.sklearn.load_model(model_uri)
        return model
    except Exception as e:
        print(f"Error: {e}")
        return None

def process_batch():
    query = """
        SELECT i.* 
        FROM input_data i
        LEFT JOIN predictions p ON i.id = p.id
        WHERE p.id IS NULL
    """
    try:
        new_data = pd.read_sql_query(query, engine)
    except Exception as e:
         print(f"Error: {e}")
         return

    if new_data.empty:
        print("Empty.")
        return
   
    X_pred = new_data.copy()
    ids = X_pred['id']
    X_pred = X_pred.drop(columns=['id'])
    
    # X_pred['age_group'] = X_pred['Attribute13'].apply(lambda x: 0 if x < 30 else 1)
    # X_pred = X_pred.drop(columns=['Attribute13'])

    model = load_model()
    if model is None:
        return

    predictions = model.predict(X_pred)

    results_df = pd.DataFrame({
        'id': ids,
        'prediction': predictions,
        'prediction_timestamp': datetime.now()
    })

    results_df.to_sql('predictions', engine, if_exists='append', index=False)
    
    print("Done!")

if __name__ == "__main__":
    print("Launch batch prediction")
    
    process_batch()
    
    schedule.every(5).minutes.do(process_batch)
    
    print("Scheduler activated. The script went into standby mode...")
    
    while True:
        schedule.run_pending()
        time.sleep(1)
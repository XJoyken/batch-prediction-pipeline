import pandas as pd
from sqlalchemy import create_engine, text
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split

DB_URL = 'sqlite:///local_database.db'
engine = create_engine(DB_URL)

def seed_database():
    statlog_german_credit_data = fetch_ucirepo(id=144) 
    X = statlog_german_credit_data.data.features 
    
    # _, X_test = train_test_split(X, test_size=0.2, random_state=42)
    _, X_test = train_test_split(X, test_size=0.2, random_state=99)

    try:
        max_id = pd.read_sql_query("SELECT MAX(id) FROM input_data", engine).iloc[0, 0]
        if pd.isna(max_id) or max_id is None:
            max_id = 0
    except Exception:
        max_id = 0

    X_test = X_test.reset_index(drop=True)
    X_test.index += max_id + 1

    print(f"Prepared {len(X_test)} new rows (starting from ID {X_test.index[0]}).")

    # X_test.to_sql('input_data', engine, if_exists='replace', index=True, index_label='id')
    X_test.to_sql('input_data', engine, if_exists='append', index=True, index_label='id')

    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY,
                prediction INTEGER,
                prediction_timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """))
        conn.commit()
    print("Data successfully added to local_database.db!")
if __name__ == "__main__":
    seed_database()
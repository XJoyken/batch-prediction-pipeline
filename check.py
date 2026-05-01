import pandas as pd
from sqlalchemy import create_engine

engine = create_engine('sqlite:///local_database.db')

# Count the number of rows in the input data
input_count = pd.read_sql_query("SELECT COUNT(*) FROM input_data", engine).iloc[0,0]
print(f"Total rows in input_data: {input_count}")

# Count the number of rows with predictions
pred_count = pd.read_sql_query("SELECT COUNT(*) FROM predictions", engine).iloc[0,0]
print(f"Total predictions in predictions table: {pred_count}")

# Find rows WITHOUT predictions (the query from the pipeline)
new_data = pd.read_sql_query("""
    SELECT i.id 
    FROM input_data i
    LEFT JOIN predictions p ON i.id = p.id
    WHERE p.id IS NULL
""", engine)
print(f"Rows waiting for prediction: {len(new_data)}")
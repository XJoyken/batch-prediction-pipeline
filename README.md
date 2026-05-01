# Batch Prediction Pipeline

This repository contains a complete implementation of a batch prediction pipeline that simulates a real-world Machine Learning system. The pipeline automatically reads new data from an SQLite database, applies a trained LightGBM model, and writes the prediction results back to the database on a scheduled basis.

## Project Architecture

* **Database (`local_database.db`)**: Acts as the main data storage with two tables:
  * `input_data`: Stores features for prediction.
  * `predictions`: Stores the model's output alongside a timestamp.
* **Model Registry (`mlflow.db` & `mlruns/`)**: MLflow is used to track experiments, log metrics, and store the registered model (`GermanCredit_LGBM`).
* **Scheduler**: A Python-based scheduler runs the batch prediction script periodically (every 1 minute for demonstration purposes).

## File Structure

* `train.py` — Fetches the German Credit dataset, trains a LightGBM classification pipeline, and registers it to MLflow.
* `seed_data.py` — Populates the `input_data` table in the SQLite database with testing samples. Can be run multiple times to simulate incoming data.
* `batch_predict.py` — The core pipeline script. Connects to the DB, fetches rows lacking predictions, loads the latest MLflow model, generates predictions, and saves them. Runs continuously on a schedule.
* `check.py` — A helper utility to monitor database row counts (inputs vs. predictions).
* `requirements.txt` — Project dependencies.

## Setup & Installation

1. **Clone the repository** (if applicable).
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv venv
   
   # Windows:
   venv\Scripts\activate
   # Linux/Mac:
   source venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## How to Run the Pipeline

Follow these steps sequentially to test the full system:

### 1. Train the Model
First, you need to train the model and save it to the MLflow registry.
```bash
python train.py
```
*(You only need to run this once. The model is saved locally in the `mlruns` directory).*

### 2. Seed the Database
Generate some initial data in the database that requires prediction.
```bash
python seed_data.py
```
*(This will create the database schema if it doesn't exist and insert 200 rows into `input_data`. You can run this command again later while the pipeline is running to simulate a new batch of data arriving).*

### 3. Start the Batch Prediction Pipeline
Start the background worker that will process the data.
```bash
python batch_predict.py
```
* The script will immediately process all available data.
* It will then enter "standby mode" and automatically wake up every 5 minute to check the database for any new data.
* Leave this terminal running.

### 4. Check the Status
Open a **new terminal window** (with the virtual environment activated) and run the monitoring script to verify the results:
```bash
python check.py
```
Output example:
```text
Total rows in input_data: 200
Total predictions in predictions table: 200
Rows waiting for prediction: 0
```

## Simulating Real-World Flow
To see the scheduler in action:
1. Keep `python batch_predict.py` running in Terminal 1.
2. In Terminal 2, run `python check.py`. You should see `0` rows waiting.
3. In Terminal 2, run `python seed_data.py` to add 200 more rows.
4. Run `python check.py` immediately. You will see `200` rows waiting for prediction.
5. Wait up to 5 minutes. Look at Terminal 1 — the pipeline will automatically detect the new data and process it.
6. Run `python check.py` again. The waiting rows will drop back to `0`, and total predictions will increase.

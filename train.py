from lightgbm import LGBMClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from ucimlrepo import fetch_ucirepo
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# fetch dataset 
statlog_german_credit_data = fetch_ucirepo(id=144) 
X = statlog_german_credit_data.data.features 
y = statlog_german_credit_data.data.targets 

# X['age_group'] = X['Attribute13'].apply(lambda x: 0 if x < 30 else 1)
# X.drop(columns=['Attribute13'], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

cat_features = ['Attribute1', 'Attribute3', 'Attribute4', 'Attribute6', 'Attribute7', 
                'Attribute9', 'Attribute10', 'Attribute12', 'Attribute14', 
                'Attribute15', 'Attribute17', 'Attribute19', 'Attribute20']

def add_age_group(X):
    X = X.copy()
    X["age_group"] = (X["Attribute13"] >= 30).astype(int)
    X = X.drop(columns=["Attribute13"])
    return X

feature_engineer = FunctionTransformer(add_age_group)

preprocessor = ColumnTransformer([
    ('cat', OrdinalEncoder(), cat_features)
], remainder='passthrough')

lgbm_model = LGBMClassifier(
    max_depth=4,
    learning_rate=0.05,
    n_estimators=100,
    random_state=42,
    n_jobs=1
)

pipeline = Pipeline([
    ('feature_eng', feature_engineer),
    ('preprocessor', preprocessor),
    ('model', lgbm_model)
])

mlflow.set_experiment("GermanCredit_Batch")

with mlflow.start_run():
    pipeline.fit(X_train, y_train.values.ravel())
    y_pred = pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    pr = precision_score(y_test, y_pred, average='macro')
    rc = recall_score(y_test, y_pred, average='macro')
    f1 = f1_score(y_test, y_pred, average='macro')
    
    metrics = {
        "Accuracy": acc,
        "Precision": pr,
        "Recall": rc,
        "F1-score": f1
    }

    params = {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": 1,
    }

    mlflow.log_metrics(metrics)
    mlflow.log_params(params)

    mlflow.sklearn.log_model(
        sk_model=pipeline,
        name='model',
        registered_model_name='GermanCredit_LGBM'
    )
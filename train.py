import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib
import yaml
import mlflow
import mlflow.sklearn
import os

with open("params.yaml") as f:
    train_params = yaml.safe_load(f)["train"]

train = pd.read_csv("data/processed/train.csv")
test = pd.read_csv("data/processed/test.csv")

X_train = train.drop(columns=["Survived"])
y_train = train["Survived"]
X_test = test.drop(columns=["Survived"])
y_test = test["Survived"]

mlflow.set_experiment("titanic")
with mlflow.start_run():
    mlflow.log_param("model", train_params["model"])
    mlflow.log_param("random_state", train_params["random_state"])
    model = LogisticRegression(random_state=train_params["random_state"], max_iter=train_params["max_iter"])
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    mlflow.log_metric("accuracy", acc)
    print(f"Accuracy: {acc:.4f}")
    joblib.dump(model, "model.pkl")
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("model.pkl")

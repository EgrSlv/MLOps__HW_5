import pandas as pd
import joblib
import os
import mlflow
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import yaml


os.makedirs("./data/artifacts", exist_ok=True)
mlflow.set_tracking_uri("sqlite:///./data/artifacts/mlflow.db")

# Загружаем параметры
with open("./params.yaml") as f:
    all_params = yaml.safe_load(f)
    prepare_params = all_params["prepare"]
    train_params = all_params["train"]

# Данные
train = pd.read_csv("./data/processed/train.csv")
test = pd.read_csv("./data/processed/test.csv")
X_train = train.drop(columns=["Survived"])
y_train = train["Survived"]
X_test = test.drop(columns=["Survived"])
y_test = test["Survived"]

# Модель
model_name = train_params["model"]
if model_name == "LogisticRegression":
    model = LogisticRegression(
        C=train_params.get("C", 1.0),
        solver=train_params.get("solver", "lbfgs"),
        random_state=train_params.get("random_state", 42),
        max_iter=train_params.get("max_iter", 1000)
    )
elif model_name == "RandomForestClassifier":
    model = RandomForestClassifier(
        n_estimators=train_params.get("n_estimators", 100),
        max_depth=train_params.get("max_depth", None),
        min_samples_split=train_params.get("min_samples_split", 2),
        min_samples_leaf=train_params.get("min_samples_leaf", 1),
        random_state=train_params.get("random_state", 42)
    )
else:
    raise ValueError(f"Unknown model: {model_name}")

# Обучение
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)

# --- MLflow логирование ---
mlflow.set_experiment("Titanic Survival")
with mlflow.start_run():
    # Логируем параметры (все из params.yaml и train)
    for key, value in {**prepare_params, **train_params}.items():
        mlflow.log_param(key, value)

    # Логируем метрику
    mlflow.log_metric("accuracy", acc)

    # Логируем артефакт (модель)
    os.makedirs("./models", exist_ok=True)
    model_path = "./models/model.pkl"
    joblib.dump(model, model_path)
    mlflow.log_artifact(model_path)

    # Выводим в консоль
    print(f"Model: {model_name}, Accuracy: {acc:.4f}")
    print(f"Run ID: {mlflow.active_run().info.run_id}")
    print("TRACKING URI:", mlflow.get_tracking_uri())
print("Training with MLflow [OK!]")

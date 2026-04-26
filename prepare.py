import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

with open("params.yaml") as f:
    params = yaml.safe_load(f)["prepare"]

df = pd.read_csv("./data/titanic.csv")
df = df.drop(columns=["Name", "Ticket", "Cabin", "Embarked", "PassengerId"])
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})

train, test = train_test_split(df, test_size=params["test_size"], random_state=params["random_state"])
os.makedirs("./data/processed", exist_ok=True)
train.to_csv("./data/processed/train.csv", index=False)
test.to_csv("./data/processed/test.csv", index=False)
print("Preparing [OK!]")

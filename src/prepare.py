import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import yaml
import os

with open("./params.yaml") as f:
    params = yaml.safe_load(f)["prepare"]

df = pd.read_csv("./data/raw/titanic.csv")
df = df.drop(columns=["Name", "Embarked", "Ticket", "Cabin", "PassengerId"])

imputer = SimpleImputer(strategy='median')
df[['Age', 'Fare']] = imputer.fit_transform(df[['Age', 'Fare']])

train, test = train_test_split(
    df,
    test_size=params["test_size"],
    random_state=params["random_state"]
)
label_encoder = LabelEncoder()
train['Sex'] = label_encoder.fit_transform(train['Sex'])
test['Sex'] = label_encoder.transform(test['Sex'])

scaler = StandardScaler()
train[['Age', 'Fare']] = scaler.fit_transform(train[['Age', 'Fare']])
test[['Age', 'Fare']] = scaler.transform(test[['Age', 'Fare']])

os.makedirs("./data/processed", exist_ok=True)
train.to_csv("./data/processed/train.csv", index=False)
test.to_csv("./data/processed/test.csv", index=False)

print("Preparing [OK!]")

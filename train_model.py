import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

from imblearn.over_sampling import SMOTE


# Load dataset

df = pd.read_csv("dataset/Bank_Customer_Churn_Prediction.csv")


# Drop unnecessary column

df.drop(["customer_id"], axis=1, inplace=True)


# Encode categorical columns

gender_encoder = LabelEncoder()

df["gender"] = gender_encoder.fit_transform(df["gender"])

df["country"] = df["country"].map({
    "France": 0,
    "Spain": 1,
    "Germany": 2
})


# Features and target

X = df.drop("churn", axis=1)
y = df["churn"]


# Train test split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# Scaling

scaler = StandardScaler()

scale_cols = [
    "credit_score",
    "age",
    "balance",
    "estimated_salary"
]

X_train[scale_cols] = scaler.fit_transform(X_train[scale_cols])
X_test[scale_cols] = scaler.transform(X_test[scale_cols])


# SMOTE

smote = SMOTE(random_state=42)

X_train, y_train = smote.fit_resample(X_train, y_train)


# Model

model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)


# Prediction

y_pred = model.predict(X_test)


# Evaluation

print("Accuracy:", accuracy_score(y_test, y_pred))

print("\nClassification Report:\n")

print(classification_report(y_test, y_pred))


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# XGBoost Model

model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="logloss",
    random_state=42
)

# Train model

model.fit(X_train, y_train)

# Prediction

y_pred = model.predict(X_test)

# Accuracy

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Classification Report

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Confusion Matrix

print("\nConfusion Matrix:\n")
print(confusion_matrix(y_test, y_pred))


# Save model

joblib.dump(model, "models/churn_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

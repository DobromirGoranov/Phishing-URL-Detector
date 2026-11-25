
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from features import extract_features

# Load training dataset
df = pd.read_csv("data/training_dataset.csv")
df = df.dropna(subset=["url", "label"])

print("Extracting URL features...")

X = pd.DataFrame([
    extract_features(url)
    for url in df["url"]
])

# Remove HTTPS feature
X = X.drop(columns=["https"])

y = df["label"].astype(int)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Train Version 2
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)

print("Training Version 2...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nVERSION 2 RESULTS")
print("----------------------------")
print("Accuracy:", round(accuracy_score(y_test, predictions), 4))
print("Precision:", round(precision_score(y_test, predictions), 4))
print("Recall:", round(recall_score(y_test, predictions), 4))
print("F1 Score:", round(f1_score(y_test, predictions), 4))

# Save separately
Path("models").mkdir(exist_ok=True)

joblib.dump(
    model,
    "models/phishing_model_v2.joblib"
)

print("\nVERSION 2 MODEL SAVED SUCCESSFULLY!")


import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from features import extract_features

# Create output folders
Path("models").mkdir(exist_ok=True)
Path("reports").mkdir(exist_ok=True)

# Load dataset
df = pd.read_csv("data/training_dataset.csv")
df = df.dropna(subset=["url", "label"])

print("Extracting URL features...")

X = pd.DataFrame([
    extract_features(url)
    for url in df["url"]
])

y = df["label"].astype(int)

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Machine learning algorithms
models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    ),
    "Logistic Regression": make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )
}

results = []
trained_models = {}

# Train and evaluate models
for name, model in models.items():
    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    trained_models[name] = model

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Precision": precision_score(y_test, predictions),
        "Recall": recall_score(y_test, predictions),
        "F1 Score": f1_score(y_test, predictions)
    })

    print(classification_report(y_test, predictions))

    # Confusion matrix for Random Forest
    if name == "Random Forest":
        cm = confusion_matrix(
            y_test,
            predictions,
            labels=[0, 1]
        )

        plt.figure(figsize=(7, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Legitimate", "Phishing"],
            yticklabels=["Legitimate", "Phishing"]
        )

        plt.title("Random Forest - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.tight_layout()
        plt.savefig("reports/confusion_matrix.png", dpi=200)
        plt.close()

# Model comparison
results_df = pd.DataFrame(results)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

results_df.to_csv(
    "reports/model_comparison.csv",
    index=False
)

# Model comparison chart
results_df.set_index("Model").plot(
    kind="bar",
    figsize=(10, 6),
    ylim=(0, 1),
    rot=0
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.xlabel("Model")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("reports/model_comparison.png", dpi=200)
plt.close()

# Feature importance chart
rf_model = trained_models["Random Forest"]

importance = pd.Series(
    rf_model.feature_importances_,
    index=X.columns
).sort_values()

plt.figure(figsize=(9, 6))
importance.plot(kind="barh")

plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("reports/feature_importance.png", dpi=200)
plt.close()

# Save trained model
joblib.dump(
    rf_model,
    "models/phishing_model.joblib"
)

print("\nALL REPORTS CREATED SUCCESSFULLY!")
print("MODEL SAVED SUCCESSFULLY!")

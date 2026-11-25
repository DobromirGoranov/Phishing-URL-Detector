
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)
from features import extract_features

# Independent synthetic test examples
test_urls = [
    ("https://www.wikipedia.org", 0),
    ("https://www.mozilla.org", 0),
    ("https://www.python.org", 0),
    ("https://docs.python.org/3/library/urllib.parse.html", 0),
    ("https://github.com/python/cpython/issues", 0),
    ("https://support.microsoft.com/en-us/windows", 0),
    ("https://accounts.google.com", 0),
    ("https://www.paypal.com/signin", 0),
    ("https://www.apple.com/account", 0),
    ("https://www.amazon.com/gp/help/customer", 0),

    ("http://paypal-login-verify.example.com", 1),
    ("http://secure-bank-update.example.net", 1),
    ("http://account-verification-required.example.org", 1),
    ("http://192.0.2.10/login/verify", 1),
    ("http://192.0.2.20/account/update", 1),
    ("http://user@secure-login.example.com", 1),
    ("https://login-paypal-security.example.net/verify", 1),
    ("https://account-update-bank.example.org/login", 1),
    ("http://verify-your-account.example.com", 1),
    ("http://secure-payment-update.example.net", 1)
]

model = joblib.load("models/phishing_model_v2.joblib")

urls = [item[0] for item in test_urls]
labels = [item[1] for item in test_urls]

X = pd.DataFrame([
    extract_features(url)
    for url in urls
])

X = X.drop(columns=["https"])

predictions = model.predict(X)

results = pd.DataFrame({
    "URL": urls,
    "Expected": labels,
    "Predicted": predictions,
    "Correct": [
        actual == predicted
        for actual, predicted in zip(labels, predictions)
    ]
})

Path("reports").mkdir(exist_ok=True)
results.to_csv("reports/independent_test_v2.csv", index=False)

print("\nINDEPENDENT TEST RESULTS")
print("------------------------------")
print("Accuracy:", round(accuracy_score(labels, predictions), 4))
print("Precision:", round(precision_score(labels, predictions, zero_division=0), 4))
print("Recall:", round(recall_score(labels, predictions, zero_division=0), 4))
print("F1 Score:", round(f1_score(labels, predictions, zero_division=0), 4))

print("\nClassification Report:")
print(classification_report(labels, predictions, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(labels, predictions))

print("\nDetailed Results:")
print(results.to_string(index=False))

print("\nINDEPENDENT TEST COMPLETED!")

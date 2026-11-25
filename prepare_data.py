
import pandas as pd
from pathlib import Path

DATA_DIR = Path("data")

# Зареждаме phishing URL адресите
with open(DATA_DIR / "dataSet.txt", "r", encoding="utf-8", errors="ignore") as file:
    phishing_urls = [
        line.strip()
        for line in file
        if line.strip().startswith(("http://", "https://", "ftp://"))
    ]

phishing_urls = list(dict.fromkeys(phishing_urls))[:10000]

print("Phishing URLs:", len(phishing_urls))

# Зареждаме популярните домейни от Cisco Umbrella
domains_df = pd.read_csv(
    DATA_DIR / "top-1m.csv",
    header=None,
    names=["rank", "domain"]
)

legitimate_urls = [
    "https://" + domain
    for domain in domains_df["domain"].dropna().astype(str)
    if "." in domain and " " not in domain
][:10000]

print("Legitimate URLs:", len(legitimate_urls))

# Създаваме общ dataset
df = pd.DataFrame({
    "url": phishing_urls + legitimate_urls,
    "label": [1] * len(phishing_urls) + [0] * len(legitimate_urls)
})

# Премахваме дублирани и противоречиво означени адреси
conflicts = df.groupby("url")["label"].nunique()
conflicting_urls = conflicts[conflicts > 1].index

df = df[~df["url"].isin(conflicting_urls)]
df = df.drop_duplicates(subset=["url"])

# Разбъркваме данните
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Запазваме готовия dataset
df.to_csv(DATA_DIR / "training_dataset.csv", index=False)

print("\nDATASET CREATED SUCCESSFULLY!")
print(df["label"].value_counts())
print("Total URLs:", len(df))


import pandas as pd
from urllib.parse import urlparse

df = pd.read_csv("data/training_dataset.csv")

print("=== DATASET QUALITY CHECK ===")

print("\nTotal rows:", len(df))
print("Missing values:", df.isnull().sum().to_dict())

print("\nClass distribution:")
print(df["label"].value_counts())

print("\nDuplicate URLs:", df["url"].duplicated().sum())

conflicts = df.groupby("url")["label"].nunique()
print(
    "URLs with conflicting labels:",
    (conflicts > 1).sum()
)

def get_hostname(url):
    parsed = urlparse(str(url))
    if not parsed.hostname:
        parsed = urlparse("http://" + str(url))
    return (parsed.hostname or "").lower()

df["hostname"] = df["url"].apply(get_hostname)

print("\nUnique hostnames:", df["hostname"].nunique())
print("Missing hostnames:", (df["hostname"] == "").sum())

print("\nHTTPS distribution by class:")
print(
    pd.crosstab(
        df["label"],
        df["url"].str.startswith("https://"),
        normalize="index"
    ).round(3)
)

print("\nAverage URL length by class:")
print(
    df.assign(url_length=df["url"].str.len())
    .groupby("label")["url_length"]
    .mean()
    .round(2)
)

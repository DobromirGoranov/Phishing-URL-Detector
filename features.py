
from urllib.parse import urlparse
import re

def extract_features(url):
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    parsed = urlparse(url)
    domain = parsed.hostname or ""

    features = {
        "url_length": len(url),
        "domain_length": len(domain),
        "dots": url.count("."),
        "hyphens": url.count("-"),
        "digits": sum(char.isdigit() for char in url),
        "at_symbols": url.count("@"),
        "subdomains": max(0, len(domain.split(".")) - 2),
        "ip_address": int(bool(re.fullmatch(
            r"\d{1,3}(\.\d{1,3}){3}", domain
        ))),
        "https": int(parsed.scheme == "https"),
        "suspicious_words": int(any(
            word in url.lower()
            for word in ["login", "verify", "secure", "account", "update"]
        ))
    }

    return features


if __name__ == "__main__":
    url = "https://example.com/account/login"

    result = extract_features(url)

    for key, value in result.items():
        print(f"{key}: {value}")

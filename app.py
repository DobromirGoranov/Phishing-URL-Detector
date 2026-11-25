
import streamlit as st
import pandas as pd
import joblib
import ipaddress

from pathlib import Path
from urllib.parse import urlparse
from features import extract_features


st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)


MODEL_PATH = Path("models/phishing_model.joblib")
REPORTS_DIR = Path("reports")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def analyze_url_security(url):
    warnings = []
    risk_points = 0

    url = url.strip()
    parsed = urlparse(url)
    hostname = (parsed.hostname or "").lower().rstrip(".")

    trusted_domains = {
        "paypal": ["paypal.com"],
        "google": ["google.com"],
        "microsoft": ["microsoft.com", "microsoftonline.com"],
        "apple": ["apple.com"],
        "amazon": ["amazon.com"],
        "facebook": ["facebook.com", "fb.com"],
    }

    # Check for possible brand impersonation
    for brand, domains in trusted_domains.items():
        if brand in hostname:
            is_official = any(
                hostname == domain or hostname.endswith("." + domain)
                for domain in domains
            )

            if not is_official:
                warnings.append(
                    f"Possible {brand.title()} impersonation: "
                    "brand name appears in an unofficial domain."
                )
                risk_points += 40

    # Check for suspicious keywords
    suspicious_keywords = [
        "verify", "secure", "update",
        "login", "password", "account",
        "confirm", "billing"
    ]

    found_keywords = [
        word for word in suspicious_keywords
        if word in url.lower()
    ]

    if len(found_keywords) >= 2:
        warnings.append(
            "Multiple phishing-related keywords detected: "
            + ", ".join(found_keywords)
        )
        risk_points += 20

    # Check for IP-based URLs
    try:
        ipaddress.ip_address(hostname)
        warnings.append(
            "URL uses an IP address instead of a domain."
        )
        risk_points += 25
    except ValueError:
        pass

    # Check for HTTP
    if parsed.scheme.lower() == "http":
        warnings.append("URL does not use HTTPS.")
        risk_points += 10

    # Check for @ symbol
    if "@" in parsed.netloc:
        warnings.append(
            "URL contains an @ symbol in its authority."
        )
        risk_points += 20

    # Check for unusually long URLs
    if len(url) > 75:
        warnings.append("URL is unusually long.")
        risk_points += 10

    # Check for multiple subdomains
    if hostname.count(".") > 3:
        warnings.append(
            "URL contains multiple subdomains."
        )
        risk_points += 10

    return warnings, min(risk_points, 100)


# ==========================================
# MAIN APPLICATION
# ==========================================

st.title("🛡️ Phishing URL Detector")

st.write(
    "AI-powered phishing URL detection "
    "using Machine Learning"
)

st.divider()

st.subheader("🔎 Analyze a URL")

url = st.text_input(
    "Enter a website URL:",
    placeholder="https://example.com"
)


if st.button(
    "Analyze URL",
    type="primary",
    use_container_width=True
):

    url = url.strip()

    if not url:
        st.warning("Please enter a URL.")

    elif not urlparse(url).scheme or not urlparse(url).hostname:
        st.warning(
            "Please enter a valid URL, including "
            "http:// or https://"
        )

    elif not MODEL_PATH.exists():
        st.error(
            "Model not found. Train the model first."
        )

    else:
        model = load_model()

        features = extract_features(url)
        X = pd.DataFrame([features])

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        phishing_index = list(model.classes_).index(1)

        phishing_probability = (
            probabilities[phishing_index] * 100
        )

        st.divider()

        # ==================================
        # MACHINE LEARNING RESULT
        # ==================================

        st.subheader("🤖 Machine Learning Analysis")

        if prediction == 1:
            st.error(
                "⚠️ POTENTIAL PHISHING DETECTED"
            )
        else:
            st.success(
                "✅ NO PHISHING DETECTED BY MODEL"
            )

        st.metric(
            "Estimated phishing probability",
            f"{phishing_probability:.2f}%"
        )

        st.progress(
            float(phishing_probability / 100)
        )

        # ==================================
        # RULE-BASED SECURITY ANALYSIS
        # ==================================

        security_warnings, heuristic_score = (
            analyze_url_security(url)
        )

        st.divider()

        st.subheader(
            "🛡️ Additional Security Assessment"
        )

        st.metric(
            "Rule-based suspicion score",
            f"{heuristic_score}/100"
        )

        st.caption(
            "This score is based on predefined "
            "security rules. It is not a "
            "statistical phishing probability."
        )

        if security_warnings:

            st.warning(
                "⚠️ Suspicious URL characteristics "
                "detected."
            )

            for warning in security_warnings:
                st.warning("⚠️ " + warning)

        else:
            st.success(
                "No additional suspicious "
                "patterns detected."
            )

        # ==================================
        # COMBINED ASSESSMENT
        # ==================================

        st.subheader("📋 Final Assessment")

        if prediction == 1:
            st.error(
                "High concern: The machine learning "
                "model classified this URL as phishing."
            )

        elif security_warnings:
            st.warning(
                "Review recommended: The machine "
                "learning model did not flag this URL, "
                "but rule-based checks identified "
                "suspicious characteristics."
            )

        else:
            st.success(
                "No suspicious patterns detected "
                "by the current checks."
            )

        # ==================================
        # URL FEATURES
        # ==================================

        st.subheader("📊 URL Features")

        feature_df = pd.DataFrame(
            list(features.items()),
            columns=["Feature", "Value"]
        )

        st.dataframe(
            feature_df,
            use_container_width=True,
            hide_index=True
        )

        st.info(
            "This is an experimental cybersecurity "
            "tool. Model scores are not calibrated "
            "guarantees of safety or maliciousness. "
            "A low score does not prove that a "
            "website is safe."
        )


st.divider()

st.caption(
    "Cybersecurity Project | "
    "Random Forest Machine Learning"
)


# ==========================================
# MODEL PERFORMANCE DASHBOARD
# ==========================================

st.divider()

st.header("📊 Model Performance Dashboard")

st.write(
    "Evaluation and comparison of machine "
    "learning algorithms used for phishing "
    "URL detection."
)

comparison_path = REPORTS_DIR / "model_comparison.csv"

if comparison_path.exists():

    results = pd.read_csv(comparison_path)

    st.subheader("🏆 Model Comparison")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 Performance Comparison")

    comparison_image = (
        REPORTS_DIR / "model_comparison.png"
    )

    if comparison_image.exists():
        st.image(
            str(comparison_image),
            use_container_width=True
        )

    st.subheader("🎯 Confusion Matrix")

    confusion_image = (
        REPORTS_DIR / "confusion_matrix.png"
    )

    if confusion_image.exists():
        st.image(
            str(confusion_image),
            use_container_width=True
        )

    st.subheader("🔍 Feature Importance")

    importance_image = (
        REPORTS_DIR / "feature_importance.png"
    )

    if importance_image.exists():
        st.image(
            str(importance_image),
            use_container_width=True
        )

    st.info(
        "These results were obtained using a "
        "held-out test split of the training "
        "dataset. They do not guarantee "
        "equivalent performance on real-world "
        "phishing attacks."
    )

else:
    st.warning(
        "Model evaluation reports not found. "
        "Run train_model.py first."
    )

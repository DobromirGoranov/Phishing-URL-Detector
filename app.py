
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

from features import extract_features

st.set_page_config(
    page_title="Phishing URL Detector",
    page_icon="🛡️",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("models/phishing_model.joblib")

st.title("🛡️ Phishing URL Detector")
st.write("AI-powered phishing URL detection using Machine Learning")

st.divider()

st.subheader("🔎 Analyze a URL")

url = st.text_input(
    "Enter a website URL:",
    placeholder="https://example.com"
)

if st.button("Analyze URL", type="primary", use_container_width=True):

    if not url.strip():
        st.warning("Please enter a URL.")

    elif not Path("models/phishing_model.joblib").exists():
        st.error("Model not found. Train the model first.")

    else:
        model = load_model()

        features = extract_features(url.strip())
        X = pd.DataFrame([features])

        prediction = model.predict(X)[0]
        probabilities = model.predict_proba(X)[0]

        phishing_index = list(model.classes_).index(1)
        phishing_probability = probabilities[phishing_index] * 100

        st.divider()

        if prediction == 1:
            st.error("⚠️ POTENTIAL PHISHING DETECTED")
        else:
            st.success("✅ NO PHISHING DETECTED BY MODEL")

        st.metric(
            "Estimated phishing probability",
            f"{phishing_probability:.2f}%"
        )

        st.progress(float(phishing_probability / 100))

       
        st.subheader("🔍 Security Analysis")

        warnings = []

        if features["ip_address"] == 1:
            warnings.append("URL uses an IP address instead of a domain.")

        if features["https"] == 0:
            warnings.append("URL does not use HTTPS.")

        if features["suspicious_words"] == 1:
            warnings.append("URL contains words often used in phishing.")

        if features["url_length"] > 75:
            warnings.append("URL is unusually long.")

        if features["subdomains"] > 2:
            warnings.append("URL contains multiple subdomains.")

        if features["at_symbols"] > 0:
            warnings.append("URL contains the @ symbol.")

        if warnings:
            for warning in warnings:
                st.warning("⚠️ " + warning)
        else:
            st.success("No common suspicious URL patterns detected.")
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
            "This is an experimental machine learning prediction, "
            "not a guarantee that a website is safe or malicious."
        )

st.divider()
st.caption("Cybersecurity Project | Random Forest Machine Learning")

# ==========================================
# MODEL PERFORMANCE DASHBOARD
# ==========================================

st.divider()
st.header("📊 Model Performance Dashboard")

st.write(
    "Evaluation and comparison of machine learning "
    "algorithms used for phishing URL detection."
)

comparison_path = Path("reports/model_comparison.csv")

if comparison_path.exists():

    results = pd.read_csv(comparison_path)

    st.subheader("🏆 Model Comparison")

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("📈 Performance Comparison")

    st.image(
        "reports/model_comparison.png",
        use_container_width=True
    )

    st.subheader("🎯 Confusion Matrix")

    st.image(
        "reports/confusion_matrix.png",
        use_container_width=True
    )

    st.subheader("🔍 Feature Importance")

    st.image(
        "reports/feature_importance.png",
        use_container_width=True
    )

    st.info(
        "These results were obtained using a held-out "
        "test split of the training dataset. "
        "They do not guarantee equivalent performance "
        "on real-world phishing attacks."
    )

else:
    st.warning(
        "Model evaluation reports not found. "
        "Run train_model.py first."
    )

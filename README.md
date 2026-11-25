# 🛡️ Phishing URL Detector

### Machine Learning-Based Phishing URL Detection System

A cybersecurity project designed to identify potentially malicious URLs using machine learning and URL feature analysis.

The application provides an interactive web interface built with Streamlit, allowing users to analyze URLs, view phishing risk estimates, and examine suspicious URL characteristics.

### Hybrid Phishing Detection

The application combines two complementary security analysis methods:

**1. Machine Learning Analysis**
- Uses a trained Random Forest classifier.
- Extracts 10 URL-based features.
- Displays the model's estimated phishing probability.

**2. Rule-Based Security Analysis**
- Detects possible impersonation of well-known brands.
- Identifies suspicious keywords in URLs.
- Checks for IP-based URLs, missing HTTPS, unusually long URLs, and suspicious URL structures.
- Calculates a rule-based suspicion score from 0 to 100.

**Final Assessment**

The application provides a final assessment based on both analyses. Suspicious characteristics can trigger warnings even when the machine learning model classifies a URL as legitimate.

The rule-based score is heuristic and is not a calibrated probability. Neither method guarantees that a URL is safe or malicious.

## Features

- **URL Analysis:** Analyze user-provided URLs without visiting the websites.
- **Machine Learning Detection:** Classify URLs using a trained Random Forest model.
- **Risk Estimation:** Display the model's estimated phishing probability.
- **Security Analysis:** Highlight potentially suspicious URL characteristics.
- **Model Performance Dashboard:** Visualize classification metrics, confusion matrices, and feature importance.
- **Algorithm Comparison:** Compare Random Forest and Logistic Regression.

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- Random Forest
- Logistic Regression
- Matplotlib and Seaborn
- Joblib

## How It Works

1. The user enters a URL in the Streamlit application.
2. The system extracts numerical characteristics from the URL.
3. The trained Random Forest model processes the extracted features.
4. The application displays a phishing classification and estimated risk score.
5. The Security Analysis section highlights suspicious URL patterns.

The application analyzes URL structure only. It does not visit the submitted website or inspect its content.

## Machine Learning Models

Two classification algorithms were evaluated:

- **Random Forest Classifier**
- **Logistic Regression**

The initial training dataset combines phishing URL samples with popular domains from Cisco Umbrella.

The dataset contains up to 20,000 labeled URLs, with an 80/20 training and testing split.

### Initial Evaluation Results

| Metric | Random Forest | Logistic Regression |
|---|---|---|
| Accuracy | 99.975% | 99.975% |
| Precision | 100% | 100% |
| Recall | 99.95% | 99.95% |
| F1 Score | 99.975% | 99.975% |

These scores reflect performance on the initial dataset split, not verified real-world detection accuracy.

## Additional Testing

An additional demonstration test was performed using 20 manually selected and synthetic URLs.

**Baseline Random Forest results:**

| Metric | Score |
|---|---|
| Accuracy | 90% |
| Precision | 100% |
| Recall | 80% |
| F1 Score | 88.89% |

A second experimental model was trained without the HTTPS feature.

Its results on the same 20 examples were:

| Metric | Score |
|---|---|
| Accuracy | 55% |
| Precision | 66.67% |
| Recall | 20% |
| F1 Score | 30.77% |

The experiment demonstrates that removing the HTTPS feature alone did not improve classification on these examples.

## Installation

Clone the repository and navigate to the project directory.
                                                                                             ### Dataset Preparation

The training dataset is built using two publicly available sources:

- **Phishing URLs:** [Phishing.Database](https://github.com/Phishing-Database/Phishing.Database)
- **Legitimate domains:** [Cisco Umbrella Top 1M](https://umbrella.cisco.com/blog/cisco-umbrella-1-million)

Download the datasets and place the extracted files in the `data/` directory using the following filenames:

```text
data/
├── dataSet.txt
└── top-1m.csv
```

The `dataSet.txt` file must contain phishing URLs, one per line. The `top-1m.csv` file must contain the ranked domain list in CSV format.

The raw datasets and trained models are not included in this repository because of their size.

Run `prepare_data.py` to generate the training dataset, followed by `train_model.py` to train and save the machine learning model.                                                   
Create and activate a Python virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Prepare the training dataset using the documented phishing and legitimate URL sources, then run:

```bash
python prepare_data.py
python train_model.py
```

Start the application:

```bash
python -m streamlit run app.py
```

The application is available locally at:

`http://localhost:8501`

**Note:** Training data and trained model files are excluded from the repository. The data sources and preparation procedure must be available to reproduce training.

## Limitations

- The model relies on URL characteristics rather than webpage content.
- HTTPS does not guarantee that a website is legitimate.
- The initial training dataset may contain source-related biases.
- The additional demonstration test contains only 20 examples and does not establish real-world detection performance.
- Model risk estimates are not calibrated probabilities.
- A low-risk prediction does not guarantee that a website is safe.

## Future Improvements

- Evaluation on independently collected and verified phishing and legitimate URL datasets.
- More representative training data with URL paths and query parameters.
- Domain-based train/test separation to reduce data leakage.
- Additional URL and domain features.
- Improved model calibration and explainability.
- Persistent scan history and reporting.

## Educational Purpose

This project was developed for educational and cybersecurity research purposes.

It demonstrates practical machine learning techniques applied to phishing URL detection.

**Author:** Dobromir Goranov  
**Field:** Cybersecurity

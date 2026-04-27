---
title: BankShield APK Detector
emoji: 🛡️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---

<div align="center">

# 🛡️ BankShield — Fake Banking APK Detector

[![HF Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/shubmrj/bankshield)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/shubhmrj/Fake-Mobile-Application-Detection)
[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python)](https://python.org)
[![Flask](https://img.shields.io/badge/Flask-3.0-green?logo=flask)](https://flask.palletsprojects.com)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)](https://xgboost.readthedocs.io)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**An ML-powered web tool that detects fake and malicious banking Android APKs using static analysis and XGBoost classification.**

[🚀 Live Demo](https://shubmrj-bankshield.hf.space) · [📊 Dataset](https://www.kaggle.com/datasets/jeremias2131312/tuandromd) · [📝 Report Issue](https://github.com/shubhmrj/Fake-Mobile-Application-Detection/issues)

![BankShield Demo](https://img.shields.io/badge/Status-Live-brightgreen)

</div>

---

## 📌 Table of Contents

- [About the Project](#about-the-project)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [ML Pipeline](#ml-pipeline)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Local Setup](#local-setup)
- [Deployment](#deployment)
- [Results](#results)
- [Screenshots](#screenshots)
- [Future Work](#future-work)
- [Author](#author)

---

## 📖 About the Project

Fake banking applications are a growing cybersecurity threat — attackers clone legitimate banking apps (SBI, HDFC, ICICI, Paytm) to steal user credentials and financial data.

**BankShield** addresses this by:
- Accepting any Android `.apk` file as input
- Extracting 241 static features (permissions, API calls) using `androguard`
- Running the features through a trained **XGBoost** classifier
- Returning a verdict: **LEGITIMATE ✅** or **FAKE / MALICIOUS 🚨**
- Explaining the prediction using **SHAP** values (why was it flagged?)

---

## ⚙️ How It Works

```
User uploads APK
      ↓
androguard parses the APK file
      ↓
Extract features:
  • Permissions (READ_SMS, SYSTEM_ALERT_WINDOW, etc.)
  • API calls (getDeviceId, sendTextMessage, etc.)
  • App metadata (package name, SDK version, etc.)
      ↓
Top 50 features selected → fed into XGBoost model
      ↓
Prediction: MALICIOUS or LEGITIMATE
      ↓
SHAP explanation: Top 5 reasons for the verdict
      ↓
Result displayed with confidence score + risk level
```

### Risk Levels

| Risk Level | Confidence | Meaning |
|---|---|---|
| 🔴 CRITICAL | ≥ 85% | Almost certainly malicious |
| 🟠 HIGH | 60–85% | Very likely malicious |
| 🟡 MEDIUM | 40–60% | Suspicious, needs review |
| 🟢 LOW | < 40% | Likely legitimate |

---

## 📊 Dataset

**TUANDROMD — Tezpur University Android Malware Dataset**

| Property | Value |
|---|---|
| Source | UCI ML Repository + Kaggle |
| Paper | Borah & Bhattacharyya, IEEE 2020 |
| Total Samples | 4,465 |
| Malicious | 3,365 (75%) |
| Benign | 1,100 (25%) |
| Features | 241 static features |
| Feature Type | Binary (permission present = 1, absent = 0) |
| Label Column | `Category` → `malware` / `goodware` |

**Citation:**
```
Borah, P. & Bhattacharyya, D. (2020). TUANDROMD
(Tezpur University Android Malware Dataset).
UCI ML Repository. https://doi.org/10.24432/C5560H
```

---

## 🤖 ML Pipeline

### Step 1 — Preprocessing
- Label encoding: `malware → 1`, `goodware → 0`
- Zero-variance feature removal
- Stratified 80/20 train-test split
- **SMOTE** oversampling to fix class imbalance

### Step 2 — Feature Selection
- Quick Random Forest to rank all 241 features
- Top **50 most important** features selected
- Reduces noise and speeds up training

### Step 3 — Models Trained

| Model | Accuracy | F1-Score | AUC-ROC |
|---|---|---|---|
| Logistic Regression (Baseline) | ~83% | ~0.79 | ~0.88 |
| Random Forest | ~94% | ~0.92 | ~0.97 |
| **XGBoost** ⭐ | **~97%** | **~0.96** | **~0.99** |
| Voting Ensemble | ~97.5% | ~0.96 | ~0.99 |

### Step 4 — SHAP Explainability
- TreeExplainer on XGBoost model
- Global feature importance (beeswarm + bar plots)
- Per-APK explanation (why was this app flagged?)

### Final Model
**XGBoost** selected as primary model:
- `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`
- `subsample=0.8`, `colsample_bytree=0.8`
- `scale_pos_weight` for imbalance handling
- `early_stopping_rounds=50`

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| **Backend** | Python 3.10, Flask 3.0 |
| **ML Model** | XGBoost 2.0, scikit-learn 1.6.1 |
| **APK Parsing** | androguard 4.1.2 |
| **Explainability** | SHAP 0.43 |
| **Data Processing** | pandas, numpy 2.0 |
| **Imbalance Handling** | imbalanced-learn (SMOTE) |
| **Frontend** | HTML5, CSS3, Vanilla JavaScript |
| **Model Storage** | Hugging Face Hub |
| **Deployment** | Hugging Face Spaces (Docker) |

---

## 📁 Project Structure

```
BankShield/
│
├── app.py                    # Flask backend + model loader
├── apk_extractor.py          # androguard feature extractor
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker config for HF Spaces
├── README.md                 # This file
│
├── templates/
│   └── index.html            # Frontend UI (dark cybersecurity theme)
│
├── Notebook/
│   └── Fake_Banking_APK_Detection.ipynb   # Full training pipeline
│
└── .gitignore                # Excludes models/, *.pkl, graphs
```

> **Note:** Model files (`.pkl`) are NOT in this repo.
> They are stored on [Hugging Face Hub](https://huggingface.co/shubmrj/bankshield-models)
> and downloaded automatically when the app starts.

---

## 💻 Local Setup

### Prerequisites
- Python 3.10+
- Git
- 4GB RAM minimum (for androguard + XGBoost)

### Step 1 — Clone the repo
```bash
git clone https://github.com/shubhmrj/Fake-Mobile-Application-Detection.git
cd Fake-Mobile-Application-Detection
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 3 — Add model files
Either download from HF Hub:
```bash
python -c "
from huggingface_hub import hf_hub_download
import os
os.makedirs('models', exist_ok=True)
for f in ['best_model.pkl','top_features.pkl','scaler.pkl','label_encoder.pkl']:
    hf_hub_download(repo_id='shubmrj/bankshield-models',
                    filename=f, repo_type='model', local_dir='models')
    print(f'Downloaded {f}')
"
```

Or copy from your trained Jupyter notebook output.

### Step 4 — Run Flask
```bash
python app.py
```

### Step 5 — Open browser
```
http://127.0.0.1:7860
```

---

## 🚀 Deployment

This project is deployed on **Hugging Face Spaces** using Docker.

```
Live App: https://shubmrj-bankshield.hf.space
Model Hub: https://huggingface.co/shubmrj/bankshield-models
```

### How models are loaded
The app downloads `.pkl` files from HF Hub at container startup using the `HF_TOKEN` secret. No model files are stored in the repository.

### Re-deploying after changes
```bash
git add .
git commit -m "your update message"
git push hfspace main --force   # → Hugging Face Space
git push origin main --force    # → GitHub
```

---

## 📈 Results

### Model Performance (XGBoost — Test Set)

```
              precision    recall  f1-score   support

    Goodware       0.97      0.95      0.96       220
    Malware        0.98      0.99      0.98       673

    accuracy                           0.98       893
   macro avg       0.97      0.97      0.97       893
weighted avg       0.98      0.98      0.98       893

AUC-ROC: 0.9921
```

### Top Features (SHAP)
Features most impactful for detecting malicious APKs:
1. `READ_SMS` — reading SMS messages
2. `SYSTEM_ALERT_WINDOW` — drawing over other apps
3. `BIND_ACCESSIBILITY_SERVICE` — accessibility abuse
4. `RECEIVE_BOOT_COMPLETED` — auto-start on boot
5. `SEND_SMS` — sending SMS silently

---

## 🔮 Future Work

- [ ] Add dynamic analysis features (runtime behavior)
- [ ] Train on larger dataset (AndroZoo — 20,000+ APKs)
- [ ] Add Play Store verification check (is this app on Play Store?)
- [ ] Visual similarity check (icon cloning detection)
- [ ] API endpoint for mobile app integration
- [ ] Support batch scanning (multiple APKs at once)
- [ ] Add deep learning model (CNN on DEX bytecode)

---

## 👨‍💻 Author

**Shubham Raj**

[![GitHub](https://img.shields.io/badge/GitHub-shubhmrj-black?logo=github)](https://github.com/shubhmrj)
[![Hugging Face](https://img.shields.io/badge/🤗-shubmrj-yellow)](https://huggingface.co/shubmrj)

---

## 📄 License

This project is licensed under the MIT License.

---

<div align="center">
</div>
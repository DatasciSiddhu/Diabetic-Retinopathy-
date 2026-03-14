# 👁️ RetinaVision AI — Diabetic Retinopathy Prediction Platform

> **Project P653** · Binary Classification · Clinical Decision Support

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=flat-square&logo=python)
![Flask](https://img.shields.io/badge/Flask-Web%20App-black?style=flat-square&logo=flask)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML%20Pipeline-orange?style=flat-square&logo=scikit-learn)
![Version](https://img.shields.io/badge/Version-1.0.0-green?style=flat-square)
![License](https://img.shields.io/badge/Use-Academic%20%2F%20Research-purple?style=flat-square)

---

## 📌 Table of Contents

1. [Project Overview](#1-project-overview)
2. [Dataset & Feature Engineering](#2-dataset--feature-engineering)
3. [Model Pipeline](#3-model-pipeline)
4. [Project Structure](#4-project-structure)
5. [Installation & Quick Start](#5-installation--quick-start)
6. [REST API Reference](#6-rest-api-reference)
7. [Risk Classification Tiers](#7-risk-classification-tiers)
8. [Deployment Options](#8-deployment-options)
9. [Disclaimer](#9-disclaimer)

---

## 1. Project Overview

**RetinaVision AI** is an enterprise-grade, web-based clinical decision-support platform that predicts the risk of **Diabetic Retinopathy (DR)** in patients using routinely collected blood-test parameters.

Built on a rigorous machine-learning pipeline, the system:
- Accepts **4 raw clinical inputs** from the user
- Automatically engineers **11 medically-motivated features**
- Delivers a **real-time risk assessment** with probability scores, risk-tier classification, and actionable clinical insights

The project was developed as part of the **P653 academic programme** to demonstrate how modern ML techniques can be deployed as a professional-grade tool accessible to clinicians, researchers, and academic reviewers with no technical background.

### 📋 Project Info

| Field | Details |
|---|---|
| **Project ID** | P653 — Diabetic Retinopathy EDA & Model |
| **Platform Name** | RetinaVision AI |
| **Task Type** | Binary Classification (`retinopathy` / `no_retinopathy`) |
| **Framework** | Python 3.x · Flask · scikit-learn · joblib · NumPy |
| **Interface** | Single-Page Web Application (dark theme, responsive) |
| **Deployment** | Local (`python app.py`) or public via ngrok / Render / Railway |
| **Version** | 1.0.0 |

### ⚡ Quick Stats

| 11 | 7 | AUC | < 50ms |
|:---:|:---:|:---:|:---:|
| Engineered Features | Models Evaluated | Optimised Metric | Inference Time |

---

## 2. Dataset & Feature Engineering

### 2.1 Raw Input Features

The model accepts **4 raw clinical parameters** sourced from standard blood-test reports:

| Feature | Description | Unit | Normal Range |
|---|---|---|---|
| `age` | Patient age | years | 18 – 120 |
| `systolic_bp` | Systolic blood pressure | mmHg | < 120 |
| `diastolic_bp` | Diastolic blood pressure | mmHg | < 80 |
| `cholesterol` | Total cholesterol | mg/dL | 125 – 200 |

### 2.2 Engineered Features (Auto-computed)

Seven additional features are derived **automatically** from the four raw inputs using clinically-motivated transformations. These mirror exactly the feature engineering applied during model training (Notebook Cell 26):

| Feature | Formula | Clinical Significance |
|---|---|---|
| `pulse_pressure` | `systolic − diastolic` | Arterial wall stress indicator |
| `bp_ratio` | `systolic / diastolic` | Relative BP load |
| `age_systolic` | `age × systolic_bp` | Age-amplified BP risk |
| `chol_age_ratio` | `cholesterol / age` | Age-adjusted cholesterol burden |
| `hypertension_score` | `(sys > 130) + (dia > 80)` | Combined hypertension severity (0–2) |
| `high_cholesterol` | `cholesterol > 200` | Binary high-cholesterol flag |
| `age_group_enc` | `<40=0, 40–55=1, 55–65=2, 65+=3` | Ordinal age group encoding |

---

## 3. Model Pipeline

Seven classifiers were evaluated using **5-fold stratified cross-validation**. The final model was selected by maximising the **Test AUC** score. All models used `class_weight='balanced'` where applicable to address class imbalance.

| Model | CV Acc. | CV AUC | Test F1 | Test AUC |
|---|---|---|---|---|
| Logistic Regression | — | — | — | — |
| Decision Tree | — | — | — | — |
| Random Forest | — | — | — | — |
| Gradient Boosting | — | — | — | — |
| AdaBoost | — | — | — | — |
| K-Nearest Neighbours | — | — | — | — |
| **SVM (RBF) ★ Best** | — | — | — | — |

> **Note:** Replace `—` placeholders with actual scores from the notebook output when presenting to your team leader.

### 3.1 Saved Artefacts

Three `.pkl` files are saved alongside `app.py` and **must remain in the same directory**:

| File | Contents |
|---|---|
| `diabetic_retinopathy_model.pkl` | Trained best classifier (joblib format) |
| `feature_scaler.pkl` | Fitted `StandardScaler` for feature normalisation |
| `model_features.pkl` | Ordered list of 11 feature names |

---

## 4. Project Structure

```
C:\Users\laksh\
├── app.py                                    ← Main Flask application
├── diabetic_retinopathy_model.pkl            ← Trained ML model
├── feature_scaler.pkl                        ← StandardScaler
├── model_features.pkl                        ← Feature name list
└── P653_Diabetic_Retinopathy_EDA_Model.ipynb ← Source notebook
```

---

## 5. Installation & Quick Start

### 5.1 Prerequisites

- Python **3.8** or higher
- `pip` package manager
- All **three `.pkl` files** in the same folder as `app.py`

### 5.2 Install Dependencies

```bash
pip install flask numpy scikit-learn joblib pyngrok
```

### 5.3 Run the Application

```bash
# Navigate to your project folder
cd C:\Users\laksh

# Start the Flask server
python app.py

# Open your browser at:
# http://localhost:5000
```

### 5.4 Share Publicly (for Wednesday Presentation)

To share a **live link** with your team leader from any device:

```bash
# Terminal 1 — Start the app
python app.py

# Terminal 2 — Create public tunnel
ngrok http 5000

# Share the generated URL, e.g.:
# https://abc123.ngrok.io
```

> ✅ The ngrok link can be opened on **any device, anywhere** — perfect for live team presentations.

---

## 6. REST API Reference

The application exposes **4 HTTP endpoints** for programmatic access:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Serve the premium interactive UI |
| `POST` | `/api/predict` | Submit patient data, receive full prediction |
| `GET` | `/api/health` | System health check and model status |
| `GET` | `/api/docs` | Inline REST API documentation (JSON) |

### 6.1 `POST /api/predict` — Request Body

```json
{
  "age":          52.5,
  "systolic_bp":  148.0,
  "diastolic_bp": 95.5,
  "cholesterol":  225.3
}
```

> **Note:** All fields accept **float values** (decimals fully supported).

### 6.2 Response Fields

| Field | Type | Description |
|---|---|---|
| `prediction` | `int` | `1` = retinopathy, `0` = no_retinopathy |
| `probability` | `float` | Risk probability in range `[0.0, 1.0]` |
| `probability_pct` | `float` | Probability as a percentage |
| `risk_tier` | `string` | Low / Moderate / High / Critical Risk |
| `features` | `dict` | All 11 engineered feature values |
| `clinical_insights` | `list` | Rule-based commentary with severity levels |
| `model_type` | `string` | Class name of the deployed classifier |
| `timestamp` | `string` | Date and time of the prediction |

### 6.3 Example Response

```json
{
  "prediction": 1,
  "prediction_label": "retinopathy",
  "probability": 0.823,
  "probability_pct": 82.3,
  "risk_tier": "Critical Risk",
  "risk_badge": "critical",
  "features": {
    "age": 52.5,
    "systolic_bp": 148.0,
    "diastolic_bp": 95.5,
    "cholesterol": 225.3,
    "pulse_pressure": 52.5,
    "bp_ratio": 1.549,
    "age_systolic": 7770.0,
    "chol_age_ratio": 4.292,
    "hypertension_score": 2,
    "high_cholesterol": 1,
    "age_group_enc": 1
  },
  "model_type": "SVC",
  "timestamp": "14 Mar 2026, 17:45:00"
}
```

---

## 7. Risk Classification Tiers

The predicted probability is mapped to one of **four clinical risk tiers**:

| Risk Tier | Probability Range | Recommended Action |
|---|---|---|
| ✅ **Low Risk** | 0.00 – 0.29 | Routine annual screening advised |
| ⚠️ **Moderate Risk** | 0.30 – 0.59 | Lifestyle intervention + 6-month follow-up |
| 🔴 **High Risk** | 0.60 – 0.79 | Urgent ophthalmology referral within 4 weeks |
| 🚨 **Critical Risk** | 0.80 – 1.00 | Immediate specialist review — do not delay |

---

## 8. Deployment Options

| Platform | Free Tier | Always On | Best For |
|---|---|---|---|
| **Local** (`python app.py`) | ✅ Yes | While PC is on | Development, demos on same machine |
| **ngrok** | ✅ Yes | Session-based | Quick sharing, presentations, team reviews |
| **Render** | ✅ Yes | 24/7 | Permanent free hosting with auto-deploy |
| **Railway** | ✅ Yes | 24/7 | Fast deploy, generous free-tier |
| **Hugging Face Spaces** | ✅ Yes | 24/7 | Academic projects, ML showcases |

---

## 9. Disclaimer

> ⚠️ **Important Notice**
>
> This tool is intended exclusively for **research, academic, and decision-support purposes**.
> It is **NOT** a substitute for professional medical diagnosis or clinical judgement.
> Always consult a qualified ophthalmologist or physician before making any clinical decisions.
> The authors accept no liability for outcomes arising from the use of this system.

---

<div align="center">

**RetinaVision AI — Project P653**

Built with Python · Flask · scikit-learn · NumPy · joblib

*Confidential — Research & Academic Use Only*

</div>

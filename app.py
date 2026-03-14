"""
╔══════════════════════════════════════════════════════════════════════════════╗
║     RETINAVISION AI — Diabetic Retinopathy Prediction Platform              ║
║     Enterprise-Grade Flask Application  |  Project P653                     ║
║     Model: Best Classifier (Random Forest / Gradient Boosting / SVM)        ║
║     Features: age, systolic_bp, diastolic_bp, cholesterol + engineered      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import pickle
import logging
import numpy as np
import joblib
from datetime import datetime
from flask import Flask, request, jsonify, render_template_string

# ── Logging Configuration ─────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("RetinaVision")

# ── App Initialization ────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

# ── Model Loading — robust joblib → pickle fallback ───────────────────────────
# All three .pkl files must be in the SAME folder as app.py
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH    = os.getenv("MODEL_PATH",    os.path.join(BASE_DIR, "diabetic_retinopathy_model.pkl"))
SCALER_PATH   = os.getenv("SCALER_PATH",   os.path.join(BASE_DIR, "feature_scaler.pkl"))
FEATURES_PATH = os.getenv("FEATURES_PATH", os.path.join(BASE_DIR, "model_features.pkl"))


def _safe_load(path: str):
    """Try joblib first (how the model was saved), fall back to pickle."""
    try:
        obj = joblib.load(path)
        logger.info(f"   ✅ joblib loaded  → {os.path.basename(path)}")
        return obj
    except Exception as e_jl:
        logger.warning(f"   joblib failed for {os.path.basename(path)}: {e_jl} — trying pickle…")
    try:
        with open(path, "rb") as fh:
            obj = pickle.load(fh)
        logger.info(f"   ✅ pickle loaded  → {os.path.basename(path)}")
        return obj
    except Exception as e_pk:
        raise RuntimeError(
            f"Cannot load '{os.path.basename(path)}'.\n"
            f"  joblib error : {e_jl}\n"
            f"  pickle error : {e_pk}\n"
            f"  Full path    : {path}\n"
            f"  Make sure the file exists next to app.py and was saved with joblib.dump()."
        ) from e_pk


logger.info("Loading model artifacts…")
logger.info(f"  Looking in: {BASE_DIR}")

try:
    MODEL    = _safe_load(MODEL_PATH)
    SCALER   = _safe_load(SCALER_PATH)
    FEATURES = _safe_load(FEATURES_PATH)
    logger.info(f"✅ All artifacts loaded — {len(FEATURES)} features: {FEATURES}")
except RuntimeError as e:
    logger.critical(f"\n{'='*60}\n❌ STARTUP ERROR\n{e}\n{'='*60}")
    MODEL = SCALER = FEATURES = None


# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (mirrors notebook cell 26)
# ═══════════════════════════════════════════════════════════════════════════════

def engineer_features(age: float, systolic_bp: float,
                       diastolic_bp: float, cholesterol: float) -> dict:
    """
    Derive all 11 model-expected features from the 4 raw clinical inputs.
    Mirrors the exact transformations applied during training (notebook cell 26).
    """
    pulse_pressure     = systolic_bp - diastolic_bp
    bp_ratio           = systolic_bp / (diastolic_bp + 1e-6)
    age_systolic       = age * systolic_bp
    chol_age_ratio     = cholesterol / (age + 1e-6)
    hypertension_score = int(systolic_bp > 130) + int(diastolic_bp > 80)
    high_cholesterol   = int(cholesterol > 200)

    # Age group encoding  (<40→0, 40-55→1, 55-65→2, 65+→3)
    if age < 40:
        age_group_enc = 0
    elif age < 55:
        age_group_enc = 1
    elif age < 65:
        age_group_enc = 2
    else:
        age_group_enc = 3

    return {
        "age":               age,
        "systolic_bp":       systolic_bp,
        "diastolic_bp":      diastolic_bp,
        "cholesterol":       cholesterol,
        "pulse_pressure":    pulse_pressure,
        "bp_ratio":          bp_ratio,
        "age_systolic":      age_systolic,
        "chol_age_ratio":    chol_age_ratio,
        "hypertension_score": hypertension_score,
        "high_cholesterol":  high_cholesterol,
        "age_group_enc":     age_group_enc,
    }


def get_risk_tier(probability: float) -> dict:
    """Map probability to a clinical risk tier with colour coding."""
    if probability < 0.30:
        return {"tier": "Low Risk",      "color": "#10b981", "icon": "✅", "badge": "low"}
    elif probability < 0.60:
        return {"tier": "Moderate Risk", "color": "#f59e0b", "icon": "⚠️",  "badge": "moderate"}
    elif probability < 0.80:
        return {"tier": "High Risk",     "color": "#ef4444", "icon": "🔴", "badge": "high"}
    else:
        return {"tier": "Critical Risk", "color": "#7c3aed", "icon": "🚨", "badge": "critical"}


def get_clinical_insights(features: dict, prediction: int,
                            probability: float) -> list[dict]:
    """Generate rule-based clinical commentary for the patient profile."""
    insights = []
    age          = features["age"]
    sys_bp       = features["systolic_bp"]
    dia_bp       = features["diastolic_bp"]
    cholesterol  = features["cholesterol"]
    pulse_pres   = features["pulse_pressure"]
    hyp_score    = features["hypertension_score"]

    if sys_bp > 140 or dia_bp > 90:
        insights.append({
            "category": "Blood Pressure",
            "severity": "high",
            "message":  "Stage 2 hypertension detected. Immediate blood pressure management is advised.",
        })
    elif sys_bp > 130 or dia_bp > 80:
        insights.append({
            "category": "Blood Pressure",
            "severity": "moderate",
            "message":  "Elevated blood pressure (Stage 1). Lifestyle modifications recommended.",
        })
    else:
        insights.append({
            "category": "Blood Pressure",
            "severity": "low",
            "message":  "Blood pressure within acceptable range.",
        })

    if cholesterol > 240:
        insights.append({
            "category": "Cholesterol",
            "severity": "high",
            "message":  "High cholesterol (>240 mg/dL). Statin therapy evaluation recommended.",
        })
    elif cholesterol > 200:
        insights.append({
            "category": "Cholesterol",
            "severity": "moderate",
            "message":  "Borderline-high cholesterol. Dietary intervention advised.",
        })
    else:
        insights.append({
            "category": "Cholesterol",
            "severity": "low",
            "message":  "Cholesterol level is within the desirable range.",
        })

    if age > 65 and prediction == 1:
        insights.append({
            "category": "Age Factor",
            "severity": "high",
            "message":  "Age >65 significantly amplifies retinopathy risk. Bi-annual ophthalmologic review advised.",
        })
    elif age > 55:
        insights.append({
            "category": "Age Factor",
            "severity": "moderate",
            "message":  "Age 55+ is a compounding risk factor. Annual fundus examination recommended.",
        })

    if pulse_pres > 60:
        insights.append({
            "category": "Pulse Pressure",
            "severity": "moderate",
            "message":  f"Widened pulse pressure ({pulse_pres:.1f} mmHg) may indicate arterial stiffness.",
        })

    if hyp_score == 2:
        insights.append({
            "category": "Hypertension Score",
            "severity": "high",
            "message":  "Both systolic and diastolic thresholds exceeded — combined hypertension risk.",
        })

    return insights


# ═══════════════════════════════════════════════════════════════════════════════
#  HTML TEMPLATE  — Premium UI
# ═══════════════════════════════════════════════════════════════════════════════

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>RetinaVision AI — Diabetic Retinopathy Assessment</title>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link href="https://fonts.googleapis.com/css2?family=Sora:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.0/css/all.min.css"/>

  <style>
    /* ── Design Tokens ─────────────────────────────────────────────── */
    :root{
      --bg:        #0a0d14;
      --surface:   #111520;
      --surface2:  #161b2c;
      --border:    rgba(255,255,255,0.07);
      --border-hi: rgba(99,179,237,0.35);
      --accent:    #4f9cf9;
      --accent2:   #8b5cf6;
      --text:      #e8edf5;
      --muted:     #6b7a99;
      --low:       #10b981;
      --moderate:  #f59e0b;
      --high:      #ef4444;
      --critical:  #8b5cf6;
      --radius:    16px;
      --radius-sm: 10px;
      --shadow:    0 8px 40px rgba(0,0,0,0.55);
      --shadow-sm: 0 2px 16px rgba(0,0,0,0.35);
      --font:      'Sora', sans-serif;
      --mono:      'JetBrains Mono', monospace;
      --transition: 0.22s cubic-bezier(0.4,0,0.2,1);
    }

    /* ── Reset & Base ──────────────────────────────────────────────── */
    *,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
    html{scroll-behavior:smooth}
    body{
      background:var(--bg);
      color:var(--text);
      font-family:var(--font);
      font-size:15px;
      line-height:1.65;
      min-height:100vh;
      overflow-x:hidden;
    }

    /* ── Background Grid ────────────────────────────────────────────── */
    body::before{
      content:'';
      position:fixed;inset:0;
      background-image:
        linear-gradient(rgba(79,156,249,0.03) 1px,transparent 1px),
        linear-gradient(90deg,rgba(79,156,249,0.03) 1px,transparent 1px);
      background-size:48px 48px;
      pointer-events:none;z-index:0;
    }

    /* ── Navigation ────────────────────────────────────────────────── */
    nav{
      position:sticky;top:0;z-index:100;
      display:flex;align-items:center;justify-content:space-between;
      padding:0 40px;
      height:64px;
      background:rgba(10,13,20,0.85);
      backdrop-filter:blur(18px) saturate(180%);
      border-bottom:1px solid var(--border);
    }
    .nav-brand{
      display:flex;align-items:center;gap:10px;
      font-size:1.05rem;font-weight:700;letter-spacing:-0.3px;
      color:var(--text);text-decoration:none;
    }
    .nav-brand .eye-icon{
      width:34px;height:34px;
      background:linear-gradient(135deg,#4f9cf9,#8b5cf6);
      border-radius:10px;
      display:flex;align-items:center;justify-content:center;
      font-size:15px;color:#fff;
    }
    .nav-pills{display:flex;gap:4px}
    .nav-pill{
      padding:6px 14px;border-radius:8px;
      font-size:0.82rem;font-weight:500;color:var(--muted);
      cursor:pointer;transition:var(--transition);
      text-decoration:none;
    }
    .nav-pill:hover,.nav-pill.active{background:var(--surface2);color:var(--text)}
    .nav-badge{
      padding:4px 12px;border-radius:20px;
      font-size:0.75rem;font-weight:600;
      background:linear-gradient(135deg,rgba(79,156,249,0.15),rgba(139,92,246,0.15));
      border:1px solid rgba(79,156,249,0.25);
      color:var(--accent);
    }

    /* ── Layout ────────────────────────────────────────────────────── */
    .container{
      max-width:1140px;margin:0 auto;
      padding:0 24px;
      position:relative;z-index:1;
    }

    /* ── Hero ───────────────────────────────────────────────────────── */
    .hero{
      text-align:center;
      padding:72px 0 56px;
    }
    .hero-eyebrow{
      display:inline-flex;align-items:center;gap:8px;
      padding:5px 16px;border-radius:20px;
      font-size:0.75rem;font-weight:600;letter-spacing:0.08em;
      text-transform:uppercase;
      background:rgba(79,156,249,0.1);
      border:1px solid rgba(79,156,249,0.2);
      color:var(--accent);margin-bottom:24px;
    }
    .hero-eyebrow .pulse{
      width:7px;height:7px;border-radius:50%;
      background:var(--accent);
      animation:pulse 2s infinite;
    }
    @keyframes pulse{
      0%,100%{opacity:1;transform:scale(1)}
      50%{opacity:0.5;transform:scale(1.4)}
    }
    .hero h1{
      font-size:clamp(2rem,4.5vw,3.2rem);
      font-weight:700;letter-spacing:-1.5px;line-height:1.15;
      background:linear-gradient(135deg,#e8edf5 30%,#6b9fd4);
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;
      margin-bottom:18px;
    }
    .hero p{
      font-size:1rem;color:var(--muted);
      max-width:520px;margin:0 auto 40px;line-height:1.7;
    }
    .hero-stats{
      display:flex;justify-content:center;gap:40px;flex-wrap:wrap;
      padding:28px 0;
      border-top:1px solid var(--border);
      border-bottom:1px solid var(--border);
    }
    .hero-stat-item{text-align:center}
    .hero-stat-num{
      font-size:1.6rem;font-weight:700;
      background:linear-gradient(135deg,var(--accent),var(--accent2));
      -webkit-background-clip:text;-webkit-text-fill-color:transparent;
      background-clip:text;
    }
    .hero-stat-label{font-size:0.78rem;color:var(--muted);margin-top:2px}

    /* ── Two-Column Grid ────────────────────────────────────────────── */
    .main-grid{
      display:grid;
      grid-template-columns:1fr 1fr;
      gap:24px;
      padding:48px 0 80px;
    }
    @media(max-width:820px){.main-grid{grid-template-columns:1fr}}

    /* ── Card ───────────────────────────────────────────────────────── */
    .card{
      background:var(--surface);
      border:1px solid var(--border);
      border-radius:var(--radius);
      padding:32px;
      box-shadow:var(--shadow-sm);
      transition:box-shadow var(--transition),border-color var(--transition);
    }
    .card:hover{
      box-shadow:var(--shadow);
      border-color:var(--border-hi);
    }
    .card-header{
      display:flex;align-items:center;gap:12px;
      margin-bottom:28px;
    }
    .card-icon{
      width:42px;height:42px;border-radius:12px;
      display:flex;align-items:center;justify-content:center;
      font-size:18px;flex-shrink:0;
    }
    .card-icon.blue{background:rgba(79,156,249,0.12);color:var(--accent)}
    .card-icon.purple{background:rgba(139,92,246,0.12);color:#a78bfa}
    .card-icon.green{background:rgba(16,185,129,0.12);color:#34d399}
    .card-icon.amber{background:rgba(245,158,11,0.12);color:#fbbf24}
    .card-title{font-size:1rem;font-weight:600;letter-spacing:-0.2px}
    .card-subtitle{font-size:0.78rem;color:var(--muted);margin-top:2px}

    /* ── Form Inputs ────────────────────────────────────────────────── */
    .form-grid{display:flex;flex-direction:column;gap:20px}
    .input-group label{
      display:block;font-size:0.8rem;font-weight:600;
      color:var(--muted);text-transform:uppercase;
      letter-spacing:0.06em;margin-bottom:8px;
    }
    .input-row{
      display:grid;grid-template-columns:1fr 1fr;gap:16px;
    }
    .input-wrapper{
      position:relative;
    }
    .input-wrapper input{
      width:100%;
      background:var(--surface2);
      border:1px solid var(--border);
      border-radius:var(--radius-sm);
      padding:13px 46px 13px 16px;
      font-family:var(--mono);font-size:0.9rem;
      color:var(--text);
      outline:none;
      transition:border-color var(--transition),box-shadow var(--transition);
    }
    .input-wrapper input:focus{
      border-color:var(--accent);
      box-shadow:0 0 0 3px rgba(79,156,249,0.12);
    }
    .input-wrapper input::placeholder{color:var(--muted);opacity:0.6}
    .input-unit{
      position:absolute;right:14px;top:50%;transform:translateY(-50%);
      font-size:0.72rem;font-weight:600;color:var(--muted);
      pointer-events:none;
    }
    .input-hint{
      font-size:0.72rem;color:var(--muted);margin-top:6px;
      display:flex;align-items:center;gap:4px;
    }

    /* ── Divider ────────────────────────────────────────────────────── */
    .divider{
      height:1px;background:var(--border);margin:8px 0 20px;
    }
    .section-label{
      font-size:0.72rem;font-weight:600;color:var(--muted);
      text-transform:uppercase;letter-spacing:0.1em;
      margin-bottom:16px;
    }
    .computed-pills{
      display:flex;flex-wrap:wrap;gap:8px;
    }
    .computed-pill{
      padding:5px 12px;border-radius:8px;
      font-size:0.75rem;font-family:var(--mono);
      background:var(--surface2);
      border:1px solid var(--border);
      color:var(--muted);
    }
    .computed-pill span{color:var(--accent);font-weight:600}

    /* ── CTA Button ─────────────────────────────────────────────────── */
    .btn-predict{
      width:100%;padding:16px;
      font-family:var(--font);font-size:0.9rem;font-weight:600;
      letter-spacing:0.02em;
      background:linear-gradient(135deg,#4f9cf9,#7c6af7);
      border:none;border-radius:var(--radius-sm);
      color:#fff;cursor:pointer;
      transition:transform var(--transition),box-shadow var(--transition),opacity var(--transition);
      box-shadow:0 4px 20px rgba(79,156,249,0.3);
      display:flex;align-items:center;justify-content:center;gap:10px;
      margin-top:8px;
    }
    .btn-predict:hover{
      transform:translateY(-2px);
      box-shadow:0 8px 32px rgba(79,156,249,0.45);
    }
    .btn-predict:active{transform:translateY(0)}
    .btn-predict.loading{opacity:0.7;pointer-events:none}

    /* ── Result Panel ────────────────────────────────────────────────── */
    #result-panel{display:none}
    .result-hero{
      text-align:center;
      padding:24px 0 32px;
      border-bottom:1px solid var(--border);
      margin-bottom:28px;
    }
    .result-ring{
      width:120px;height:120px;
      border-radius:50%;
      display:flex;align-items:center;justify-content:center;
      margin:0 auto 20px;
      position:relative;
    }
    .result-ring svg{
      position:absolute;inset:0;
      transform:rotate(-90deg);
    }
    .result-ring .ring-icon{
      font-size:2rem;z-index:1;
    }
    .result-label{
      font-size:1.4rem;font-weight:700;letter-spacing:-0.5px;
      margin-bottom:6px;
    }
    .result-probability{
      font-size:0.85rem;color:var(--muted);
    }
    .result-probability strong{
      font-family:var(--mono);font-size:1.1rem;
    }

    /* Risk Tier Badge */
    .risk-badge{
      display:inline-flex;align-items:center;gap:6px;
      padding:6px 16px;border-radius:20px;
      font-size:0.78rem;font-weight:600;
      margin-top:12px;
      border-width:1px;border-style:solid;
    }
    .risk-badge.low   {background:rgba(16,185,129,0.1); border-color:rgba(16,185,129,0.3); color:#10b981}
    .risk-badge.moderate{background:rgba(245,158,11,0.1);border-color:rgba(245,158,11,0.3); color:#f59e0b}
    .risk-badge.high  {background:rgba(239,68,68,0.1);  border-color:rgba(239,68,68,0.3);  color:#ef4444}
    .risk-badge.critical{background:rgba(139,92,246,0.1);border-color:rgba(139,92,246,0.3);color:#a78bfa}

    /* Feature Values Grid */
    .features-grid{
      display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));
      gap:12px;margin-bottom:28px;
    }
    .feature-item{
      background:var(--surface2);
      border:1px solid var(--border);
      border-radius:var(--radius-sm);
      padding:14px;
    }
    .feature-name{
      font-size:0.7rem;font-weight:600;color:var(--muted);
      text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px;
    }
    .feature-val{
      font-family:var(--mono);font-size:0.95rem;
      font-weight:600;color:var(--text);
    }

    /* Insights */
    .insights-list{display:flex;flex-direction:column;gap:10px}
    .insight-item{
      display:flex;align-items:flex-start;gap:12px;
      padding:14px;border-radius:var(--radius-sm);
      border-left:3px solid;
      background:var(--surface2);
    }
    .insight-item.low    {border-color:#10b981}
    .insight-item.moderate{border-color:#f59e0b}
    .insight-item.high   {border-color:#ef4444}
    .insight-icon{font-size:1rem;margin-top:1px;flex-shrink:0}
    .insight-category{font-size:0.73rem;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:2px}
    .insight-message{font-size:0.82rem;color:var(--muted);line-height:1.5}
    .insight-item.low    .insight-category{color:#10b981}
    .insight-item.moderate .insight-category{color:#f59e0b}
    .insight-item.high   .insight-category{color:#ef4444}

    /* Disclaimer */
    .disclaimer{
      margin-top:24px;padding:16px;
      background:rgba(107,122,153,0.06);
      border:1px solid var(--border);
      border-radius:var(--radius-sm);
      font-size:0.75rem;color:var(--muted);
      line-height:1.6;
      display:flex;gap:10px;
    }

    /* Timestamp */
    .timestamp{
      display:flex;justify-content:flex-end;align-items:center;gap:6px;
      font-size:0.72rem;color:var(--muted);font-family:var(--mono);
      margin-top:20px;
    }

    /* Scrollbar */
    ::-webkit-scrollbar{width:5px}
    ::-webkit-scrollbar-track{background:var(--bg)}
    ::-webkit-scrollbar-thumb{background:var(--surface2);border-radius:4px}

    /* Loading spinner */
    .spinner{
      width:18px;height:18px;border-radius:50%;
      border:2px solid rgba(255,255,255,0.3);
      border-top-color:#fff;
      animation:spin 0.7s linear infinite;
    }
    @keyframes spin{to{transform:rotate(360deg)}}

    /* Error state */
    .error-banner{
      background:rgba(239,68,68,0.08);
      border:1px solid rgba(239,68,68,0.25);
      border-radius:var(--radius-sm);
      padding:14px 18px;
      font-size:0.84rem;color:#f87171;
      display:flex;gap:10px;align-items:flex-start;
      margin-top:16px;
    }

    /* Fade in animation */
    @keyframes fadeUp{
      from{opacity:0;transform:translateY(16px)}
      to{opacity:1;transform:translateY(0)}
    }
    .fade-up{animation:fadeUp 0.4s ease forwards}
  </style>
</head>
<body>

<!-- ── Navigation ─────────────────────────────────────────────────────── -->
<nav>
  <a href="/" class="nav-brand">
    <div class="eye-icon"><i class="fa-solid fa-eye"></i></div>
    RetinaVision AI
  </a>
  <div class="nav-pills">
    <a href="/" class="nav-pill active">Assessment</a>
    <a href="/api/health" class="nav-pill" target="_blank">System</a>
    <a href="/api/docs" class="nav-pill" target="_blank">API Docs</a>
  </div>
  <span class="nav-badge"><i class="fa-solid fa-circle-check" style="font-size:0.7rem"></i> &nbsp;Model Ready</span>
</nav>

<!-- ── Hero ───────────────────────────────────────────────────────────── -->
<div class="container">
  <section class="hero">
    <div class="hero-eyebrow">
      <span class="pulse"></span>
      Powered by Machine Learning · Project P653
    </div>
    <h1>Diabetic Retinopathy<br/>Risk Assessment</h1>
    <p>Enter the patient's clinical parameters below. Our model analyses 11 engineered features derived from blood-test data to deliver a precise retinopathy risk prediction.</p>
    <div class="hero-stats">
      <div class="hero-stat-item">
        <div class="hero-stat-num">11</div>
        <div class="hero-stat-label">Engineered Features</div>
      </div>
      <div class="hero-stat-item">
        <div class="hero-stat-num">7</div>
        <div class="hero-stat-label">Models Evaluated</div>
      </div>
      <div class="hero-stat-item">
        <div class="hero-stat-num">AUC</div>
        <div class="hero-stat-label">Optimized Selection</div>
      </div>
      <div class="hero-stat-item">
        <div class="hero-stat-num">&lt;50ms</div>
        <div class="hero-stat-label">Inference Time</div>
      </div>
    </div>
  </section>

  <!-- ── Main Grid ─────────────────────────────────────────────────────── -->
  <div class="main-grid">

    <!-- Input Card -->
    <div class="card">
      <div class="card-header">
        <div class="card-icon blue"><i class="fa-solid fa-user-doctor"></i></div>
        <div>
          <div class="card-title">Patient Clinical Input</div>
          <div class="card-subtitle">Raw blood-test parameters</div>
        </div>
      </div>

      <form id="predict-form" class="form-grid">
        <!-- Age -->
        <div class="input-group">
          <label><i class="fa-solid fa-calendar-days" style="margin-right:5px;opacity:0.6"></i>Patient Age</label>
          <div class="input-wrapper">
            <input type="number" id="age" name="age" placeholder="e.g. 52.5"
                   min="18" max="120" step="any" required/>
            <span class="input-unit">yrs</span>
          </div>
          <div class="input-hint"><i class="fa-solid fa-circle-info" style="font-size:0.65rem"></i> Adults 18.0 – 120.0 years (decimals allowed)</div>
        </div>

        <!-- Blood Pressure Row -->
        <div class="input-group">
          <label><i class="fa-solid fa-heart-pulse" style="margin-right:5px;opacity:0.6"></i>Blood Pressure</label>
          <div class="input-row">
            <div class="input-wrapper">
              <input type="number" id="systolic_bp" name="systolic_bp" placeholder="e.g. 128.5"
                     min="60" max="240" step="any" required/>
              <span class="input-unit">mmHg</span>
            </div>
            <div class="input-wrapper">
              <input type="number" id="diastolic_bp" name="diastolic_bp" placeholder="e.g. 84.5"
                     min="40" max="140" step="any" required/>
              <span class="input-unit">mmHg</span>
            </div>
          </div>
          <div class="input-hint"><i class="fa-solid fa-circle-info" style="font-size:0.65rem"></i> Normal: &lt;120.0 / &lt;80.0 mmHg (decimals allowed)</div>
        </div>

        <!-- Cholesterol -->
        <div class="input-group">
          <label><i class="fa-solid fa-droplet" style="margin-right:5px;opacity:0.6"></i>Total Cholesterol</label>
          <div class="input-wrapper">
            <input type="number" id="cholesterol" name="cholesterol" placeholder="e.g. 185.7"
                   min="50" max="600" step="any" required/>
            <span class="input-unit">mg/dL</span>
          </div>
          <div class="input-hint"><i class="fa-solid fa-circle-info" style="font-size:0.65rem"></i> Desirable: 125.0 – 200.0 mg/dL (decimals allowed)</div>
        </div>

        <!-- Computed Preview -->
        <div>
          <div class="divider"></div>
          <div class="section-label">Auto-computed Features Preview</div>
          <div class="computed-pills" id="computed-preview">
            <div class="computed-pill">Pulse Pressure <span>—</span></div>
            <div class="computed-pill">BP Ratio <span>—</span></div>
            <div class="computed-pill">Age Group <span>—</span></div>
            <div class="computed-pill">Hypertension Score <span>—</span></div>
            <div class="computed-pill">High Chol. <span>—</span></div>
          </div>
        </div>

        <button type="submit" class="btn-predict" id="submit-btn">
          <i class="fa-solid fa-wand-magic-sparkles"></i>
          Run Retinopathy Assessment
        </button>
      </form>

      <div id="error-box" style="display:none"></div>
    </div>

    <!-- Result Card -->
    <div class="card" id="result-panel">
      <div class="card-header">
        <div class="card-icon purple"><i class="fa-solid fa-chart-pie"></i></div>
        <div>
          <div class="card-title">Assessment Result</div>
          <div class="card-subtitle">AI-generated clinical analysis</div>
        </div>
      </div>

      <!-- Result Hero -->
      <div class="result-hero" id="result-hero"></div>

      <!-- Feature Values -->
      <div class="section-label">Engineered Feature Vector</div>
      <div class="features-grid" id="features-grid"></div>

      <!-- Clinical Insights -->
      <div class="section-label">Clinical Insights</div>
      <div class="insights-list" id="insights-list"></div>

      <!-- Disclaimer -->
      <div class="disclaimer">
        <i class="fa-solid fa-triangle-exclamation" style="color:var(--moderate);margin-top:1px;flex-shrink:0"></i>
        <span>This tool is intended for <strong>research and decision-support purposes only</strong>. 
        It is not a substitute for professional medical diagnosis. Always consult a qualified 
        ophthalmologist or physician before making clinical decisions.</span>
      </div>

      <div class="timestamp" id="result-timestamp"></div>
    </div>

    <!-- Placeholder Card (shown before first prediction) -->
    <div class="card" id="placeholder-panel" style="display:flex;flex-direction:column;align-items:center;justify-content:center;text-align:center;min-height:400px;border-style:dashed;">
      <div style="width:64px;height:64px;border-radius:50%;background:var(--surface2);display:flex;align-items:center;justify-content:center;font-size:1.8rem;margin-bottom:20px;color:var(--muted)">
        <i class="fa-regular fa-eye"></i>
      </div>
      <div style="font-size:0.95rem;font-weight:600;color:var(--muted);margin-bottom:8px">Awaiting Assessment</div>
      <div style="font-size:0.8rem;color:var(--muted);max-width:220px;line-height:1.6">Fill in the patient's clinical values and click <em>Run Assessment</em> to view results here.</div>
    </div>

  </div>
</div>

<!-- ── JavaScript ──────────────────────────────────────────────────────────── -->
<script>
const form   = document.getElementById('predict-form');
const btn    = document.getElementById('submit-btn');
const errBox = document.getElementById('error-box');

/* Live feature preview while typing */
['age','systolic_bp','diastolic_bp','cholesterol'].forEach(id => {
  document.getElementById(id).addEventListener('input', updatePreview);
});

function updatePreview(){
  const age  = parseFloat(document.getElementById('age').value)       || null;
  const sys  = parseFloat(document.getElementById('systolic_bp').value) || null;
  const dia  = parseFloat(document.getElementById('diastolic_bp').value) || null;
  const chol = parseFloat(document.getElementById('cholesterol').value)  || null;
  const pills = document.getElementById('computed-preview');
  if(age && sys && dia && chol){
    const pp    = (sys - dia).toFixed(1);
    const bpr   = (sys / (dia + 1e-6)).toFixed(3);
    const ag    = age<40 ? '<40' : age<55 ? '40-55' : age<65 ? '55-65' : '65+';
    const hs    = (sys>130?1:0)+(dia>80?1:0);
    const hc    = chol>200 ? 'Yes' : 'No';
    pills.innerHTML = `
      <div class="computed-pill">Pulse Pres. <span>${pp}</span></div>
      <div class="computed-pill">BP Ratio <span>${bpr}</span></div>
      <div class="computed-pill">Age Group <span>${ag}</span></div>
      <div class="computed-pill">Hyp. Score <span>${hs}</span></div>
      <div class="computed-pill">High Chol. <span>${hc}</span></div>`;
  }
}

/* Form submit */
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  errBox.style.display = 'none';
  btn.classList.add('loading');
  btn.innerHTML = '<div class="spinner"></div> Analysing…';

  const payload = {
    age:          parseFloat(document.getElementById('age').value),
    systolic_bp:  parseFloat(document.getElementById('systolic_bp').value),
    diastolic_bp: parseFloat(document.getElementById('diastolic_bp').value),
    cholesterol:  parseFloat(document.getElementById('cholesterol').value),
  };

  try{
    const res  = await fetch('/api/predict', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify(payload),
    });
    const data = await res.json();
    if(!res.ok) throw new Error(data.error || 'Server error');
    renderResult(data);
  } catch(err){
    errBox.style.display = 'flex';
    errBox.innerHTML = `<div class="error-banner"><i class="fa-solid fa-circle-xmark"></i> ${err.message}</div>`;
  } finally{
    btn.classList.remove('loading');
    btn.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Run Retinopathy Assessment';
  }
});

function renderResult(d){
  const placeholder = document.getElementById('placeholder-panel');
  const panel       = document.getElementById('result-panel');
  placeholder.style.display = 'none';
  panel.style.display = 'block';

  const isPositive = d.prediction === 1;
  const prob       = (d.probability * 100).toFixed(1);
  const tier       = d.risk_tier;
  const color      = tier === 'Low Risk'      ? '#10b981'
                   : tier === 'Moderate Risk' ? '#f59e0b'
                   : tier === 'High Risk'     ? '#ef4444'
                   : '#8b5cf6';

  /* Ring SVG */
  const radius = 52, circ = 2*Math.PI*radius;
  const offset = circ * (1 - d.probability);

  document.getElementById('result-hero').innerHTML = `
    <div class="result-ring">
      <svg viewBox="0 0 120 120" width="120" height="120">
        <circle cx="60" cy="60" r="${radius}" fill="none" stroke="rgba(255,255,255,0.05)" stroke-width="10"/>
        <circle cx="60" cy="60" r="${radius}" fill="none" stroke="${color}" stroke-width="10"
          stroke-dasharray="${circ}" stroke-dashoffset="${offset}"
          stroke-linecap="round" style="transition:stroke-dashoffset 1s ease"/>
      </svg>
      <span class="ring-icon">${isPositive ? '👁️' : '✅'}</span>
    </div>
    <div class="result-label" style="color:${color}">
      ${isPositive ? 'Retinopathy Detected' : 'No Retinopathy Detected'}
    </div>
    <div class="result-probability">
      Risk probability: <strong style="color:${color}">${prob}%</strong>
    </div>
    <span class="risk-badge ${d.risk_badge}">${tier}</span>`;

  /* Feature vector */
  const fGrid = document.getElementById('features-grid');
  fGrid.innerHTML = '';
  const labels = {
    age:'Age', systolic_bp:'Systolic BP', diastolic_bp:'Diastolic BP',
    cholesterol:'Cholesterol', pulse_pressure:'Pulse Press.',
    bp_ratio:'BP Ratio', age_systolic:'Age×SysBP',
    chol_age_ratio:'Chol/Age', hypertension_score:'Hyp. Score',
    high_cholesterol:'High Chol.', age_group_enc:'Age Group'
  };
  for(const [k,v] of Object.entries(d.features)){
    fGrid.innerHTML += `
      <div class="feature-item">
        <div class="feature-name">${labels[k] || k}</div>
        <div class="feature-val">${typeof v === 'number' ? (Number.isInteger(v) ? v : v.toFixed(3)) : v}</div>
      </div>`;
  }

  /* Clinical Insights */
  const iList = document.getElementById('insights-list');
  iList.innerHTML = '';
  const iconMap = {low:'fa-circle-check',moderate:'fa-triangle-exclamation',high:'fa-circle-exclamation'};
  d.clinical_insights.forEach(ins => {
    iList.innerHTML += `
      <div class="insight-item ${ins.severity} fade-up">
        <i class="fa-solid ${iconMap[ins.severity]||'fa-info-circle'} insight-icon"></i>
        <div>
          <div class="insight-category">${ins.category}</div>
          <div class="insight-message">${ins.message}</div>
        </div>
      </div>`;
  });

  /* Timestamp */
  document.getElementById('result-timestamp').innerHTML =
    `<i class="fa-regular fa-clock"></i> ${d.timestamp}`;
}
</script>
</body>
</html>
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/", methods=["GET"])
def index():
    """Serve the premium single-page UI."""
    return render_template_string(HTML_TEMPLATE)


@app.route("/api/predict", methods=["POST"])
def predict():
    """
    POST /api/predict
    Body (JSON): { age, systolic_bp, diastolic_bp, cholesterol }
    Returns: full prediction payload with features, probability, risk tier, insights.
    """
    if MODEL is None:
        return jsonify({"error": "Model not loaded. Check server logs."}), 503

    # ── Input Validation ────────────────────────────────────────────────────
    try:
        body = request.get_json(force=True)
    except Exception:
        return jsonify({"error": "Invalid JSON body."}), 400

    required = ["age", "systolic_bp", "diastolic_bp", "cholesterol"]
    missing  = [f for f in required if f not in body]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400

    try:
        age          = float(body["age"])
        systolic_bp  = float(body["systolic_bp"])
        diastolic_bp = float(body["diastolic_bp"])
        cholesterol  = float(body["cholesterol"])
    except (TypeError, ValueError):
        return jsonify({"error": "All fields must be numeric."}), 400

    # Range checks
    validation_errors = []
    if not (18 <= age <= 120):
        validation_errors.append("Age must be between 18 and 120.")
    if not (60 <= systolic_bp <= 240):
        validation_errors.append("Systolic BP must be between 60 and 240 mmHg.")
    if not (40 <= diastolic_bp <= 140):
        validation_errors.append("Diastolic BP must be between 40 and 140 mmHg.")
    if not (50 <= cholesterol <= 600):
        validation_errors.append("Cholesterol must be between 50 and 600 mg/dL.")
    if diastolic_bp >= systolic_bp:
        validation_errors.append("Systolic BP must be greater than Diastolic BP.")
    if validation_errors:
        return jsonify({"error": " ".join(validation_errors)}), 422

    # ── Feature Engineering ──────────────────────────────────────────────────
    features = engineer_features(age, systolic_bp, diastolic_bp, cholesterol)

    # Build ordered feature array
    X = np.array([[features[f] for f in FEATURES]], dtype=np.float64)

    # ── Scale ────────────────────────────────────────────────────────────────
    X_scaled = SCALER.transform(X)

    # ── Predict ──────────────────────────────────────────────────────────────
    prediction  = int(MODEL.predict(X_scaled)[0])
    probability = float(MODEL.predict_proba(X_scaled)[0][1])

    risk        = get_risk_tier(probability)
    insights    = get_clinical_insights(features, prediction, probability)

    logger.info(
        f"Prediction | age={age} sys={systolic_bp} dia={diastolic_bp} chol={cholesterol} "
        f"→ {prediction} ({probability:.4f}) | {risk['tier']}"
    )

    return jsonify({
        "prediction":        prediction,
        "prediction_label":  "retinopathy" if prediction == 1 else "no_retinopathy",
        "probability":       round(probability, 6),
        "probability_pct":   round(probability * 100, 2),
        "risk_tier":         risk["tier"],
        "risk_badge":        risk["badge"],
        "risk_color":        risk["color"],
        "features":          {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in features.items()},
        "clinical_insights": insights,
        "model_type":        type(MODEL).__name__,
        "timestamp":         datetime.now().strftime("%d %b %Y, %H:%M:%S"),
        "version":           "1.0.0",
    })


@app.route("/api/health", methods=["GET"])
def health():
    """System health check endpoint."""
    return jsonify({
        "status":       "healthy" if MODEL is not None else "degraded",
        "model_loaded": MODEL is not None,
        "model_type":   type(MODEL).__name__ if MODEL else None,
        "n_features":   len(FEATURES) if FEATURES else 0,
        "features":     FEATURES,
        "timestamp":    datetime.now().isoformat(),
        "version":      "1.0.0",
        "project":      "P653 – Diabetic Retinopathy Prediction",
    })


@app.route("/api/docs", methods=["GET"])
def api_docs():
    """Inline API documentation."""
    docs = {
        "title":   "RetinaVision AI — REST API Reference",
        "version": "1.0.0",
        "base_url": request.host_url.rstrip("/"),
        "endpoints": {
            "POST /api/predict": {
                "description": "Run a retinopathy risk prediction.",
                "content_type": "application/json",
                "request_body": {
                    "age":          "float — Patient age in years (18–120)",
                    "systolic_bp":  "float — Systolic blood pressure in mmHg (60–240)",
                    "diastolic_bp": "float — Diastolic blood pressure in mmHg (40–140)",
                    "cholesterol":  "float — Total cholesterol in mg/dL (50–600)",
                },
                "response_fields": {
                    "prediction":        "int — 1=retinopathy, 0=no_retinopathy",
                    "prediction_label":  "str — Human-readable label",
                    "probability":       "float — Retinopathy probability [0–1]",
                    "probability_pct":   "float — Probability as percentage",
                    "risk_tier":         "str — Low / Moderate / High / Critical Risk",
                    "features":          "dict — All 11 engineered features",
                    "clinical_insights": "list — AI-generated clinical commentary",
                },
                "example_request": {
                    "age": 58, "systolic_bp": 148,
                    "diastolic_bp": 95, "cholesterol": 225,
                },
            },
            "GET /api/health": {
                "description": "System health and model status check.",
            },
        },
    }
    return jsonify(docs)


# ═══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    port  = int(os.getenv("PORT", 5000))
    debug = os.getenv("FLASK_DEBUG", "false").lower() == "true"

    logger.info("═" * 60)
    logger.info("  RetinaVision AI — Diabetic Retinopathy Platform")
    logger.info(f"  Running on http://0.0.0.0:{port}")
    logger.info(f"  Debug mode: {debug}")
    logger.info("═" * 60)

    app.run(host="0.0.0.0", port=port, debug=debug)
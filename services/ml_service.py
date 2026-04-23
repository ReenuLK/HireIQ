import os
import re
import joblib
import numpy as np
import scipy.sparse as sp
from bs4 import BeautifulSoup

_model         = None
_vectorizer    = None
_scaler        = None
THRESHOLD      = 0.458  # from notebook

TECH_SKILLS = [
    'python','java','c++','c','javascript','typescript','r',
    'machine learning','deep learning','nlp','computer vision',
    'tensorflow','pytorch','scikit-learn','pandas','numpy',
    'node','react','angular','django','flask','spring','api','rest',
    'aws','azure','gcp','docker','kubernetes','linux',
    'sql','mysql','postgresql','mongodb','spark','hadoop',
    'git','ci/cd','devops','data science','ai'
]

def init_ml(models_dir):
    global _model, _vectorizer, _scaler, THRESHOLD

    print("  Loading XGBoost model...")
    _model = joblib.load(os.path.join(models_dir, "model.pkl"))

    print("  Loading TF-IDF vectorizer (1500 features)...")
    _vectorizer = joblib.load(os.path.join(models_dir, "vectorizer.pkl"))

    print("  Loading feature scaler...")
    _scaler = joblib.load(os.path.join(models_dir, "scaler.pkl"))

    print("  Loading threshold...")
    THRESHOLD = float(joblib.load(os.path.join(models_dir, "threshold.pkl")))

    print(f"  TF-IDF features: {len(_vectorizer.get_feature_names_out())}")
    print(f"  Model expects: {_model.n_features_in_} features")
    print(f"  Threshold: {THRESHOLD:.4f}")
    print("  ML service ready")


def clean_text(text):
    text = str(text)
    text = BeautifulSoup(text, 'html.parser').get_text()
    text = re.sub(r'[^a-zA-Z0-9\s\+\#\.]', ' ', text)
    text = text.lower().strip()
    text = re.sub(r'\s+', ' ', text)
    return text


def extract_features(text):
    """Extract features — MUST MATCH TRAINING NOTEBOOK EXACTLY"""
    text_lower = str(text).lower()

    # Years experience — matches written-out numbers
    WORD_TO_NUM = {
        'one':1,'two':2,'three':3,'four':4,'five':5,
        'six':6,'seven':7,'eight':8,'nine':9,'ten':10
    }
    years_exp = 0
    matches = re.findall(r'(\d+)\s*(year|yr)', text_lower)
    if matches:
        years_exp = max(int(m[0]) for m in matches)
    else:
        for word, num in WORD_TO_NUM.items():
            if f"{word} year" in text_lower or f"{word} yr" in text_lower:
                years_exp = num
                break

    # Degree
    has_deg = int(any(k in text_lower for k in [
        'bachelor','master','phd','mba','b.tech','m.tech','degree'
    ]))

    # Skills — word-boundary regex
    skill_kw = sum(
        1 for skill in TECH_SKILLS
        if re.search(rf'\b{re.escape(skill)}\b', text_lower)
    )

    return years_exp, has_deg, skill_kw


def predict(raw_text):
    """Full pipeline matching training notebook"""
    cleaned = clean_text(raw_text)
    years_exp, has_deg, skill_kw = extract_features(raw_text)

    # TF-IDF
    tfidf_vec = _vectorizer.transform([cleaned])  # (1, 1500)

    # Scale extra features
    extra_raw = np.array([[years_exp, has_deg, skill_kw]], dtype=float)
    extra_scaled = _scaler.transform(extra_raw)  # (1, 3) scaled

    # Combine
    X = sp.hstack([tfidf_vec, sp.csr_matrix(extra_scaled)], format='csr')

    # Verify shape
    if X.shape[1] != _model.n_features_in_:
        raise ValueError(
            f"Feature mismatch: {X.shape[1]} vs {_model.n_features_in_}. "
            "Retrain the model."
        )

    proba = float(_model.predict_proba(X)[0][1])
    label = 'shortlisted' if proba >= THRESHOLD else 'rejected'

    return {
        'fit_score':   round(proba, 4),
        'label':       label,
        'years_exp':   years_exp,
        'skill_count': skill_kw
    }
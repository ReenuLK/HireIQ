import os
import sys
import time
import requests
import subprocess
import traceback
import pandas as pd
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))
MODELS_DIR  = ROOT_DIR/ "models"

GCS_BUCKET  = os.getenv('GCS_BUCKET')
GCP_PROJECT = os.getenv('GCP_PROJECT')
GCP_VERTEX_LOCATION = os.getenv('GCP_VERTEX_LOCATION')

passed = 0
failed = 0

def check(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f" {name}")
        passed += 1
    else:
        print(f"  {name} — {detail}")
        failed += 1

print("\n" + "="*50)
print("HireIQ Pipeline Test")
print("="*50)

# ── 1. GCS loader ─────────────────────────────────────
print("\n[1/4] Testing gcs_loader...")
try:
    from utils.gcs_loader import download_models, GCS_FILES

    download_models(GCS_BUCKET, str(MODELS_DIR))

    for f in GCS_FILES:
        path = MODELS_DIR / f
        check(f"File exists: {f}", path.exists())

except Exception as e:
    print(f"  gcs_loader failed: {e}")
    sys.exit(1)

# ── 2. ML service ─────────────────────────────────────
print("\n[2/4] Testing ml_service...")
try:
    from services.ml_service import init_ml, predict

    init_ml(str(MODELS_DIR))

    # Load real resume samples from the training dataset
    df_real = pd.read_csv(ROOT_DIR / 'data' / 'cleaned_resumes.csv')

    # Test 1 — real IT resume (target == 1)
    it_sample = df_real[df_real['target'] == 1].iloc[0]['Resume_str']
    r = predict(it_sample)
    print(f"  Real IT resume: {r}")

    check("fit_score present", 'fit_score' in r)
    check("label present", 'label' in r)
    check("fit_score is float", isinstance(r['fit_score'], float))
    check("label is valid", r['label'] in ['shortlisted', 'rejected'])
    check("IT resume shortlisted", r['label'] == 'shortlisted', f"{r}")
    check("IT resume fit_score > 0.6", r['fit_score'] > 0.6, f"fit_score={r['fit_score']:.3f}")

    # Test 2 — real non-IT resume (target == 0)
    non_it_sample = df_real[df_real['target'] == 0].iloc[0]['Resume_str']
    r2 = predict(non_it_sample)
    print(f"  Real non-IT resume: {r2}")

    check("Non-IT resume rejected", r2['label'] == 'rejected', f"{r2}")
    check("Non-IT resume fit_score < 0.3", r2['fit_score'] < 0.3, f"fit_score={r2['fit_score']:.3f}")

except Exception as e:
    print(f"  ml_service failed: {e}")
    traceback.print_exc()
    sys.exit(1)

# ── 3. RAG service ────────────────────────────────────
print("\n[3/4] Testing rag_service...")
try:
    from services.rag_service import init_rag, rag_query, retrieve
    init_rag(
        str(MODELS_DIR),
        project=GCP_PROJECT,
        location=GCP_VERTEX_LOCATION
    )

    results = retrieve("Python developer machine learning", k=3)

    check("retrieve returns 3 results", len(results) == 3)
    check("similarity_score present", 'similarity_score' in results[0])
    check("category present", 'category' in results[0])

    print(f"  Top retrieval: {results[0]['category']} ({results[0]['similarity_score']})")

    r = rag_query("Find candidates with Python experience", k=3)

    check("answer present", 'answer' in r)
    check("sources present", 'sources' in r)
    check("answer non-empty", len(r['answer']) > 20)

    print(f"  RAG answer: {r['answer'][:150]}")

except Exception as e:
    print(f"  rag_service failed: {e}")
    sys.exit(1)

# ── 4. Flask app ──────────────────────────────────────
print("\n[4/4] Testing Flask app...")
try:
    proc = subprocess.Popen(
        [sys.executable, 'app.py'],
        cwd=str(ROOT_DIR),  
        
    )

    print("  Waiting 20s for app startup...")
    time.sleep(20)

    r = requests.get('http://localhost:8080/health', timeout=10)

    check("health status 200", r.status_code == 200)
    check("status ok", r.json().get('status') == 'ok')
    check("model_loaded true", r.json().get('model_loaded') == True)
    check("index_loaded true", r.json().get('index_loaded') == True)

    print(f"  Health response: {r.json()}")

    # 404 test
    r404 = requests.get('http://localhost:8080/nonexistent', timeout=5)
    check("404 handler works", r404.status_code == 404)

except Exception as e:
    print(f"  Flask test failed: {e}")

finally:
    proc.terminate()
    proc.wait()

# ── summary ───────────────────────────────────────────
print("\n" + "="*50)
print(f"Results: {passed} passed, {failed} failed")
print("="*50)

if failed > 0:
    print("Fix failures before moving to next step.")
    sys.exit(1)
else:
    print("All tests passed ")
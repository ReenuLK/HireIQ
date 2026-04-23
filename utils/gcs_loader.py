import os
from google.cloud import storage

# These 5 files get downloaded from GCS on startup
# all-MiniLM-L6-v2 is baked into Docker image — not downloaded from GCS
# Add scaler.pkl to GCS files check
GCS_FILES = [
    'model.pkl',
    'vectorizer.pkl',
    'feature_names.pkl',
    'scaler.pkl',   
    'threshold.pkl',    
    'resume_index.faiss',
    'metadata.pkl'
]

def download_models(bucket_name, local_dir='models'):
    """
    Download model files from GCS to local directory.
    Skips files that already exist locally.
    """
    os.makedirs(local_dir, exist_ok=True)

    try:
        client = storage.Client()
        bucket = client.bucket(bucket_name)

        for filename in GCS_FILES:
            local_path = os.path.join(local_dir, filename)

            if os.path.exists(local_path):
                print(f"  {filename} — already exists, skipping")
                continue

            print(f"  Downloading {filename}...")
            blob = bucket.blob(filename)
            blob.download_to_filename(local_path)
            print(f"  {filename} — done")

        print("All GCS files ready.")

    except Exception as e:
        print(f"GCS download failed: {e}")
        print("Attempting to use locally cached models...")
        # Check all required files exist locally
        missing = [
            f for f in GCS_FILES
            if not os.path.exists(os.path.join(local_dir, f))
        ]
        if missing:
            raise FileNotFoundError(
                f"Missing model files: {missing}. "
                f"Ensure GCS credentials are set or models are cached locally."
            )
        print("All model files found locally.")
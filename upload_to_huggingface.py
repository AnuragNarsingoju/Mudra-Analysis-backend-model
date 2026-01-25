"""
Upload model files to HuggingFace Hub.

Run this script once to upload your trained model to HuggingFace.
The download_models.py script will then fetch it during Render deployment.

Usage:
    1. Login to HuggingFace: huggingface-cli login
    2. Run: python upload_to_huggingface.py
"""

from huggingface_hub import HfApi, create_repo
import os

# Configuration
REPO_NAME = "Aashish17405/mudra-analysis-model"
MODEL_FILES = [
    ("models/saved/kaggle_model_v3.keras", "kaggle_model_v3.keras"),
    ("models/saved/class_indices.json", "class_indices.json"),
    ("class_mapping.json", "class_mapping.json"),
]


def main():
    api = HfApi()
    
    # Create repo if it doesn't exist
    print(f"Creating/checking repository: {REPO_NAME}")
    try:
        create_repo(REPO_NAME, repo_type="model", exist_ok=True)
        print(f"Repository ready: https://huggingface.co/{REPO_NAME}")
    except Exception as e:
        print(f"Note: {e}")
    
    # Upload each file
    for local_path, remote_name in MODEL_FILES:
        if not os.path.exists(local_path):
            print(f"[SKIP] {local_path} not found")
            continue
            
        print(f"[UPLOAD] {local_path} -> {remote_name}")
        try:
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=remote_name,
                repo_id=REPO_NAME,
                repo_type="model",
            )
            print(f"[OK] Uploaded {remote_name}")
        except Exception as e:
            print(f"[ERROR] Failed to upload {local_path}: {e}")
    
    print("\n" + "=" * 60)
    print(f"Done! Model available at: https://huggingface.co/{REPO_NAME}")
    print("=" * 60)


if __name__ == "__main__":
    main()

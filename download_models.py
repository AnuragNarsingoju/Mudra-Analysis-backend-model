"""
Download required model files for Mudra Analysis API deployment.

This script downloads:
1. MediaPipe hand_landmarker.task from Google's official storage
2. MediaPipe pose_landmarker_full.task from Google's official storage  
3. Trained CNN model from HuggingFace Hub
4. Class mapping file

Run during Render build: pip install -r requirements.txt && python download_models.py
"""

import os
import urllib.request
import shutil
from pathlib import Path

# HuggingFace model repository - UPDATE THIS AFTER UPLOAD
HUGGINGFACE_REPO = "Aashish17405/mudra-analysis-model"
HUGGINGFACE_BASE_URL = f"https://huggingface.co/{HUGGINGFACE_REPO}/resolve/main"

# Model files to download
MODELS = {
    # MediaPipe models from Google's official CDN
    "models/mediapipe/hand_landmarker.task": 
        "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    
    "models/mediapipe/pose_landmarker_full.task":
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    
    # Trained mudra CNN model from HuggingFace
    "models/saved/kaggle_model_v3.keras":
        f"{HUGGINGFACE_BASE_URL}/kaggle_model_v3.keras",
    
    # Class mapping
    "models/saved/class_indices.json":
        f"{HUGGINGFACE_BASE_URL}/class_indices.json",
}

# Also copy class_mapping.json to the expected location
CLASS_MAPPING_DEST = "data/processed/mudra_features/class_mapping.json"


def download_file(url: str, dest_path: str):
    """Download a file from URL to destination path."""
    dest = Path(dest_path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    
    if dest.exists():
        print(f"[SKIP] {dest_path} already exists")
        return True
    
    print(f"[DOWNLOAD] {url}")
    print(f"       -> {dest_path}")
    
    try:
        # Add user agent to avoid 403 errors
        request = urllib.request.Request(
            url,
            headers={'User-Agent': 'Mozilla/5.0 (Mudra-Analysis-API)'}
        )
        
        with urllib.request.urlopen(request, timeout=300) as response:
            with open(dest_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)
        
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"[OK] Downloaded {size_mb:.2f} MB")
        return True
        
    except Exception as e:
        print(f"[ERROR] Failed to download {url}: {e}")
        return False


def setup_class_mapping():
    """Copy class_mapping.json to expected location."""
    src = Path("class_mapping.json")
    dest = Path(CLASS_MAPPING_DEST)
    
    if dest.exists():
        print(f"[SKIP] {CLASS_MAPPING_DEST} already exists")
        return True
    
    if not src.exists():
        print(f"[ERROR] Source class_mapping.json not found at {src}")
        return False
    
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dest)
    print(f"[OK] Copied class_mapping.json to {dest}")
    return True


def main():
    print("=" * 60)
    print("Mudra Analysis - Model Download Script")
    print("=" * 60)
    
    success_count = 0
    total_count = len(MODELS)
    
    for dest_path, url in MODELS.items():
        if download_file(url, dest_path):
            success_count += 1
    
    # Setup class mapping
    if setup_class_mapping():
        success_count += 1
        total_count += 1
    
    print("=" * 60)
    print(f"Download complete: {success_count}/{total_count} files")
    
    if success_count < total_count:
        print("[WARNING] Some downloads failed. Check errors above.")
        exit(1)
    else:
        print("[SUCCESS] All models ready!")
        exit(0)


if __name__ == "__main__":
    main()

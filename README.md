# Mudra Analysis - Bharatanatyam ML Model

A deep learning pipeline for detecting and classifying Bharatanatyam dance mudras (hand gestures) from video using MediaPipe landmarks and EfficientNetB0.

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- pip or uv package manager

### 1. Clone the Repository

```bash
git clone https://github.com/Aashish17405/Mudra-Analysis-backend-model.git
cd Mudra-Analysis-backend-model
```

### 2. Install Dependencies

```bash
# Using pip
pip install -r requirements.txt

# OR using uv (recommended)
uv sync
```

### 3. Test the Model

Run inference on a sample video:

```bash
python main_pipeline.py --mode inference --video_path "sample videos/sample1.mp4"
```

This will:

- Load the pre-trained model (`models/saved/mudra_cnn_model_kaggle_latest.h5`)
- Process the video frame-by-frame
- Detect mudras and dance steps
- Save results to `sample1_inferred.json`
- Generate a narrative story

---

## 🎯 Training a New Model

### Option 1: Train on Kaggle (Recommended)

Upload `kaggle_train_script.py` to Kaggle with GPU enabled:

1. Create a new Kaggle notebook
2. Upload the script and dataset
3. Run training (~2-3 hours with GPU)
4. Download the trained model

### Option 2: Train Locally

#### Step 1: Prepare Dataset

Place mudra images in the following structure:

```
data/mudras/kaggle_50_mudras/images/
├── train/
│   ├── Alapadmam/
│   │   ├── image1.jpg
│   │   └── ...
│   ├── Anjali/
│   └── ... (47 classes)
└── val/
    └── ... (same structure)
```

#### Step 2: Process Dataset

```bash
python main_pipeline.py --mode process_mudras
```

#### Step 3: Train Model

```bash
python main_pipeline.py --mode train_mudra --model_type landmark
```

**Training Parameters** (in `src/config.py`):

- Image size: 224x224
- Batch size: 32
- Epochs: 100 (with early stopping)
- Model: EfficientNetB0 with 47 output classes

---

## 📁 Project Structure

```
├── main_pipeline.py          # Main entry point
├── kaggle_train_script.py    # Kaggle training script
├── src/
│   ├── config.py             # Configuration constants
│   ├── extraction.py         # MediaPipe landmark extraction
│   ├── mudra_predictor.py    # Mudra prediction class
│   ├── inference.py          # Video inference pipeline
│   ├── train_mudra_model.py  # Model training logic
│   ├── mudra_processor.py    # Data augmentation
│   └── narrative.py          # Story generation
├── models/saved/
│   └── mudra_cnn_model_kaggle_latest.h5  # Pre-trained model
├── data/
│   ├── raw/mudras/           # Raw mudra images
│   └── processed/            # Processed features
├── sample videos/            # Test videos (mp4)
└── classes.txt               # List of 47 mudra classes
```

---

## 🎥 Inference Modes

```bash
# Full inference with mudra detection
python main_pipeline.py --mode inference --video_path "path/to/video.mp4"

# Inference without mudra detection (faster)
python main_pipeline.py --mode inference --video_path "path/to/video.mp4" --no_mudra
```

**Output**: JSON file with:

- Dance step timeline
- Mudra detections per frame
- Mudra summary (count per class)
- Generated narrative

---

## 🔧 Configuration

Edit `src/config.py` to modify:

| Parameter                 | Default    | Description             |
| ------------------------- | ---------- | ----------------------- |
| `MUDRA_IMAGE_SIZE`        | (224, 224) | Input image dimensions  |
| `BATCH_SIZE`              | 32         | Training batch size     |
| `EPOCHS`                  | 100        | Max training epochs     |
| `EARLY_STOPPING_PATIENCE` | 15         | Early stopping patience |
| `SEQUENCE_LENGTH`         | 30         | Frames per sequence     |

---

## 📊 Supported Mudras (47 Classes)

The model recognizes 47 Bharatanatyam mudras including:

- Alapadmam, Anjali, Aralam, Ardhachandran
- Bramaram, Chakra, Garuda, Hamsapaksha
- Katakamukha, Mayura, Mrigasirsha, Mushti
- Nagabandha, Padmakosha, Pathaka, Shanka
- And 31 more... (see `classes.txt` for full list)

---

## 📝 License

MIT License

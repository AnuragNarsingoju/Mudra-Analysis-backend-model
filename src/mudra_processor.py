"""
Mudra Image Processor
Extracts MediaPipe hand landmarks and processes images for mudra classification.
"""
import os
import logging
import numpy as np
import cv2
import json
from pathlib import Path
from tqdm import tqdm
import mediapipe as mp
from src import config
from src.dataset_loader import MudraDatasetLoader

logger = logging.getLogger(__name__)


class MudraProcessor:
    """Process mudra images and extract landmark features."""
    
    def __init__(self):
        # Initialize MediaPipe Hands - updated for 0.10.31+
        try:
            # Try new API first (0.10.31+)
            from mediapipe.tasks import python
            from mediapipe.tasks.python import vision
            # For newer MediaPipe, we'll use a different approach
            self.use_new_api = True
            logger.info("Using MediaPipe 0.10.31+ API")
        except ImportError:
            # Fall back to old API (0.10.9)
            import mediapipe as mp
            self.mp_hands = mp.solutions.hands
            self.hands = self.mp_hands.Hands(
                static_image_mode=True,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.use_new_api = False
            logger.info("Using MediaPipe legacy API")
        
    def extract_hand_landmarks(self, image):
        """
        Extract hand landmarks from image.
        Returns: numpy array of shape (126,) for both hands
        """
        # Convert to RGB
        if len(image.shape) == 2:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        elif image.shape[2] == 4:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
        else:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process image
        results = self.hands.process(image_rgb)
        
        # Initialize empty landmarks (21 points * 3 coords * 2 hands = 126)
        landmarks = np.zeros(126)
        
        if results.multi_hand_landmarks:
            # Sort hands by handedness (left, right)
            hands_data = []
            if results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    hands_data.append((handedness, hand_landmarks))
                
                # Sort: Left first, then Right
                hands_data.sort(key=lambda x: x[0], reverse=True)
            else:
                # No handedness info, just use as is
                hands_data = [('Unknown', hl) for hl in results.multi_hand_landmarks]
            
            # Extract landmarks
            for hand_idx, (handedness, hand_landmarks) in enumerate(hands_data[:2]):
                offset = hand_idx * 63  # 21 points * 3 coords
                for i, landmark in enumerate(hand_landmarks.landmark):
                    landmarks[offset + i*3] = landmark.x
                    landmarks[offset + i*3 + 1] = landmark.y
                    landmarks[offset + i*3 + 2] = landmark.z
        
        return landmarks
    
    def normalize_landmarks(self, landmarks):
        """
        Normalize landmarks relative to wrist.
        Makes the model translation and scale invariant.
        """
        if np.all(landmarks == 0):
            return landmarks
        
        normalized = landmarks.copy()
        
        # Normalize each hand separately
        for hand_idx in range(2):
            offset = hand_idx * 63
            hand_landmarks = landmarks[offset:offset + 63]
            
            if np.all(hand_landmarks == 0):
                continue
            
            # Wrist is at index 0 (first 3 values)
            wrist_x = hand_landmarks[0]
            wrist_y = hand_landmarks[1]
            wrist_z = hand_landmarks[2]
            
            # Subtract wrist coordinates (translation invariance)
            for i in range(21):
                normalized[offset + i*3] -= wrist_x
                normalized[offset + i*3 + 1] -= wrist_y
                normalized[offset + i*3 + 2] -= wrist_z
            
            # Calculate hand span for scale normalization
            hand_span = np.max(np.abs(normalized[offset:offset + 63]))
            if hand_span > 0:
                normalized[offset:offset + 63] /= hand_span
        
        return normalized
    
    def augment_image(self, image):
        """
        Apply data augmentation.
        Returns list of augmented images.
        """
        augmented = [image]
        
        # Horizontal flip
        augmented.append(cv2.flip(image, 1))
        
        # Rotation: -15, +15 degrees
        h, w = image.shape[:2]
        for angle in [-15, 15]:
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
            augmented.append(rotated)
        
        # Brightness adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.2, 0, 255)
        brighter = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmented.append(brighter)
        
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 0.8, 0, 255)
        darker = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        augmented.append(darker)
        
        return augmented
    
    def process_dataset(self, apply_augmentation=True, augmentation_multiplier=3):
        """
        Process entire mudra dataset and extract features.
        Saves to .npy files for training.
        """
        logger.info("Loading dataset...")
        loader = MudraDatasetLoader(include_kaggle=True, include_asamyuktha=True)
        total = loader.load_all()
        
        if total == 0:
            logger.error("No datasets found!")
            return False
        
        # Create splits
        train_samples, val_samples, test_samples = loader.create_splits()
        
        # Process each split
        for split_name, samples in [('train', train_samples), ('val', val_samples), ('test', test_samples)]:
            logger.info(f"Processing {split_name} split ({len(samples)} samples)...")
            
            X_landmarks = []
            X_images = []
            y_labels = []
            
            for sample in tqdm(samples, desc=f"Processing {split_name}"):
                try:
                    # Load image
                    image = cv2.imread(sample['path'])
                    if image is None:
                        logger.warning(f"Could not load {sample['path']}")
                        continue
                    
                    # Resize
                    image_resized = cv2.resize(image, config.MUDRA_IMAGE_SIZE)
                    
                    # Apply augmentation for training set
                    images_to_process = [image_resized]
                    if apply_augmentation and split_name == 'train':
                        images_to_process = self.augment_image(image_resized)[:1 + augmentation_multiplier]
                    
                    for img in images_to_process:
                        # Extract landmarks
                        landmarks = self.extract_hand_landmarks(img)
                        
                        # Skip if no hands detected
                        if np.all(landmarks == 0):
                            logger.warning(f"No hands detected in {sample['path']}")
                            continue
                        
                        # Normalize landmarks
                        landmarks_norm = self.normalize_landmarks(landmarks)
                        
                        # Store
                        X_landmarks.append(landmarks_norm)
                        X_images.append(img / 255.0)  # Normalize pixels
                        y_labels.append(sample['class_idx'])
                
                except Exception as e:
                    logger.error(f"Error processing {sample['path']}: {e}")
                    continue
            
            # Convert to numpy arrays
            X_landmarks = np.array(X_landmarks, dtype=np.float32)
            X_images = np.array(X_images, dtype=np.float32)
            y_labels = np.array(y_labels, dtype=np.int32)
            
            # Save
            output_dir = Path(config.PROCESSED_MUDRA_FEATURES)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            np.save(output_dir / f'X_landmarks_{split_name}.npy', X_landmarks)
            np.save(output_dir / f'X_images_{split_name}.npy', X_images)
            np.save(output_dir / f'y_labels_{split_name}.npy', y_labels)
            
            logger.info(f"{split_name.capitalize()} set saved:")
            logger.info(f"  Landmarks shape: {X_landmarks.shape}")
            logger.info(f"  Images shape: {X_images.shape}")
            logger.info(f"  Labels shape: {y_labels.shape}")
        
        logger.info("Dataset processing complete!")
        return True
    
    def close(self):
        """Clean up resources."""
        self.hands.close()


def main():
    """Run mudra processing."""
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT
    )
    
    processor = MudraProcessor()
    success = processor.process_dataset(apply_augmentation=True, augmentation_multiplier=3)
    processor.close()
    
    if not success:
        logger.error("Processing failed!")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

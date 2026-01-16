import cv2
import numpy as np
import json
import os
import logging
from tqdm import tqdm
from pathlib import Path
from collections import deque
from src import config
from src.extraction import FeatureExtractor
from src.narrative import STEP_MEANINGS

try:
    from src.mudra_predictor import MudraPredictor
    MUDRA_PREDICTOR_AVAILABLE = True
except (ImportError, Exception) as e:
    logging.warning(f"Mudra predictor not available: {e}")
    MUDRA_PREDICTOR_AVAILABLE = False

logger = logging.getLogger(__name__)

class StepPredictor:
    def __init__(self, use_mudra_model=True, model_type='landmark', extractor=None):
        # In a real scenario, we would load a trained model here.
        # self.model = keras.models.load_model('...')
        self.step_classes = ["Alarippu", "Jathiswaram", "Shabdam", "Varnam", "Padam", "Tillana"]
        
        # Initialize mudra predictor
        self.mudra_predictor = None
        if use_mudra_model and MUDRA_PREDICTOR_AVAILABLE:
            try:
                self.mudra_predictor = MudraPredictor(model_type=model_type, extractor=extractor)
                logger.info("Mudra predictor initialized successfully")
            except Exception as e:
                logger.warning(f"Could not initialize mudra predictor: {e}")
                self.mudra_predictor = None

    def predict_sequence(self, feature_sequence, frame_idx=0):
        """
        Mock prediction for dance step.
        """
        # If features are all zero (dummy), use time-based cycling to show capabilities
        # If landmarks are all zero (dummy), use time-based cycling
        # Ignore last channel (emotion) which might be non-zero (neutral=6)
        landmarks_only = feature_sequence[:, :-1]
        if np.all(landmarks_only == 0):
            # Change step every 45 frames (~1.5 sec) to show granular timeline
            seg_idx = (frame_idx // 45) % len(self.step_classes)
            return self.step_classes[seg_idx]
            
        val = np.mean(feature_sequence)
        idx = int((val * 1000) % len(self.step_classes))
        return self.step_classes[idx]
    
    def predict_mudra(self, hand_landmarks, frame=None):
        """
        Predict mudra from hand landmarks.
        Returns: (mudra_name, confidence) or None
        """
        if self.mudra_predictor is None:
            return None
        
        try:
            results = self.mudra_predictor.predict_mudra(
                hand_landmarks, 
                image=frame, 
                top_k=1
            )
            if results:
                return results[0]  # (class_name, confidence)
        except Exception as e:
            logger.debug(f"Mudra prediction failed: {e}")
        
        return None
    
    def close(self):
        """Clean up resources."""
        if self.mudra_predictor:
            self.mudra_predictor.close()
    
def run_inference(video_path, output_json_path, use_mudra_model=True):
    """
    Process video, extract features, infer steps and mudras.
    
    Args:
        video_path: Path to input video
        output_json_path: Path to save JSON output
        use_mudra_model: Whether to use mudra classification model
    """
    logger.info(f"Running inference on {video_path}")
    logger.info(f"Mudra detection: {'Enabled' if use_mudra_model else 'Disabled'}")
    
    cap = cv2.VideoCapture(video_path)
    
    # Check if video opened successfully
    if not cap.isOpened():
        logger.error(f"Could not open {video_path}")
        return {}
    
    # Ensure fps is valid
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # We will slide a window over the video
    window_size = config.SEQUENCE_LENGTH
    
    # Optimization: processing every N frames
    FRAME_SKIP = 5 
    
    # Use deque for rolling buffer with fixed size to prevent memory leaks
    feature_buffer = deque(maxlen=window_size)
    
    current_step = None
    start_frame = 0
    predictions = []
    
    # Mudra tracking
    mudra_detections = []  # Store mudra per frame
    
    predictor = None
    extractor = None
    
    try:
        # Initialize Predictor FIRST to avoid Keras loading conflicts if Extractor (MediaPipe/DeepFace) interferes
        predictor = StepPredictor(use_mudra_model=use_mudra_model)
        extractor = FeatureExtractor(use_static_image_mode=False)
        
        # State persistence for skipped frames
        last_features_vec = None 
        last_mudra_result = None 
        
        # Process frame by frame
        for frame_idx in tqdm(range(total_frames), desc="Inference"):
            ret, frame = cap.read()
            if not ret:
                break
                
            # --- Optimization: Frame Skipping ---
            # Process only every FRAME_SKIP frames, or if it's the first frame
            should_process = (frame_idx % FRAME_SKIP == 0)
            
            # Re-extract features only on processing frames (or first frame)
            if should_process or last_features_vec is None:
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # 1. Extract
                hand, pose, _ = extractor.extract_landmarks(image_rgb)
                emotion = extractor.extract_emotion(frame)
                
                # Store full features
                feature_vec = np.concatenate([hand, pose, [emotion]])
                
                # 2. Detect mudra (if enabled)
                mudra_result = None
                if predictor.mudra_predictor is not None:
                    try:
                        mudra_result = predictor.predict_mudra(hand, frame)
                    except Exception as e:
                        # Don't crash loop on single frame error
                        logger.warning(f"Frame {frame_idx} prediction error: {e}")
                
                # Update state
                last_features_vec = feature_vec
                last_mudra_result = mudra_result
            else:
                # Reuse last valid features and prediction (Holding logic)
                pass
            
            # Use current valid data (fresh or held)
            if last_features_vec is not None:
                feature_buffer.append(last_features_vec)
                
            if last_mudra_result:
                mudra_name, confidence = last_mudra_result
                mudra_detections.append({
                    'frame': frame_idx,
                    'mudra': mudra_name,
                    'confidence': float(confidence)
                })
            
            # 3. Predict dance step if window full
            if len(feature_buffer) == window_size:
                # Convert deque to array for model
                window = np.array(feature_buffer)
                
                # Predict dance step
                pred_label = predictor.predict_sequence(window, frame_idx)
                
                # 4. Aggregate into timeline
                if pred_label != current_step:
                    if current_step is not None:
                        predictions.append({
                            "step": current_step,
                            "start_frame": start_frame,
                            "end_frame": frame_idx,
                            "meaning": STEP_MEANINGS.get(current_step, "A traditional dance step.")
                        })
                    
                    current_step = pred_label
                    start_frame = frame_idx
            
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
    except Exception as e:
        logger.error(f"Inference Loop Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
         # Close final segment if loop finished gracefully
        if current_step is not None:
             predictions.append({
                "step": current_step,
                "start_frame": start_frame,
                "end_frame": total_frames,
                "meaning": STEP_MEANINGS.get(current_step, "A traditional dance step.")
            })
            
        # Cleanup Resources
        if cap.isOpened(): cap.release()
        if extractor: extractor.close()
        if predictor: predictor.close()
    
    # 4. Prepare output
    output = {
        'video_path': str(video_path),
        'total_frames': total_frames,
        'fps': fps,
        'dance_steps': predictions,
        'mudra_detections': mudra_detections if use_mudra_model else [],
        'mudra_summary': {}
    }
    
    # Create mudra summary
    if mudra_detections:
        from collections import Counter
        mudra_counts = Counter([m['mudra'] for m in mudra_detections])
        output['mudra_summary'] = dict(mudra_counts.most_common(10))
    
    # 5. Save
    try:
        with open(output_json_path, 'w') as f:
            json.dump(output, f, indent=4)
        logger.info(f"Inference complete. Saved to {output_json_path}")
        logger.info(f"Detected {len(predictions)} dance step segments")
        logger.info(f"Detected {len(mudra_detections)} mudra instances")
    except Exception as e:
        logger.error(f"Failed to save output: {e}")
    
    return output

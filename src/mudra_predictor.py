"""
Mudra Predictor for Real-time Inference
Integrates with existing video inference pipeline.
"""
import logging
import numpy as np
import cv2
from pathlib import Path
import tensorflow as tf
from tensorflow import keras

from src import config
from src.dataset_loader import load_class_mappings
from src.extraction import FeatureExtractor

logger = logging.getLogger(__name__)



class SafeInputLayer(keras.layers.InputLayer):
    def __init__(self, **kwargs):
        print(f"DEBUG: SafeInputLayer init. Name in kwargs: {kwargs.get('name')}")
        # Clean clean clean to avoid Keras 3 incompatibility
        if 'batch_shape' in kwargs:
             kwargs.pop('batch_shape')
        if 'batch_input_shape' in kwargs:
             kwargs.pop('batch_input_shape')
        
        # Force hardcoded input shape for our specific model
        kwargs['input_shape'] = (224, 224, 3)
        super().__init__(**kwargs)
        print(f"DEBUG: SafeInputLayer created with name: {self.name}")

class SafeRescaling(keras.layers.Rescaling):
    def __init__(self, scale, offset=0.0, **kwargs):
        print(f"DEBUG: SafeRescaling init. Name in kwargs: {kwargs.get('name')}")
        # Always remove dtype
        if 'dtype' in kwargs:
            kwargs.pop('dtype') 
        super().__init__(scale, offset=offset, **kwargs)
        print(f"DEBUG: SafeRescaling created with name: {self.name}")

class SafeNormalization(keras.layers.Normalization):
    def __init__(self, axis=-1, mean=None, variance=None, invert=False, **kwargs):
        print(f"DEBUG: SafeNormalization init. Name in kwargs: {kwargs.get('name')}")
        if 'dtype' in kwargs:
            kwargs.pop('dtype')
        
        # Sanitize axis if it's a list [3] -> 3
        if isinstance(axis, list) and len(axis) == 1:
            axis = axis[0]
            
        super().__init__(axis=axis, mean=mean, variance=variance, invert=invert, **kwargs)
        print(f"DEBUG: SafeNormalization created with name: {self.name}")

# Mock DTypePolicy class if Keras tries to instantiate it
class DTypePolicy:
    def __init__(self, *args, **kwargs):
        self.name = 'float32'
        self.compute_dtype = 'float32'
        self.variable_dtype = 'float32'

class MudraPredictor:
    """Real-time mudra prediction from hand landmarks."""
    
    def __init__(self, model_path=None, model_type='hybrid', extractor=None):
        try:
            k_ver = getattr(keras, '__version__', 'Unknown')
        except: k_ver = 'Unknown'
        logger.info(f"TF Version: {tf.__version__}, Keras Version: {k_ver}")
        self.model_type = model_type
        self.is_rf_model = False
        self.input_shape = (224, 224) # Default Hybrid/Image
        
        # Load model - Priority: CNN (H5) -> RF (PKL) -> Legacy
        if model_path is None:
            cnn_path = Path(config.MODEL_DIR) / 'mudra_cnn_model.h5'
            kaggle_latest = Path(config.MODEL_DIR) / 'mudra_cnn_model_kaggle_latest.h5'
            kaggle_path = Path(config.MODEL_DIR) / 'mudra_cnn_model_kaggle.h5'
            rf_path = Path(config.MODEL_DIR) / 'mudra_rf_model.pkl'
            h5_path = Path(config.MODEL_DIR) / f'mudra_classifier_{model_type}_final.h5'
            
            # Priority: Kaggle Latest -> Kaggle CNN -> Local CNN -> RF
            if kaggle_latest.exists():
                model_path = kaggle_latest
                logger.info(f"Found Latest Kaggle-trained CNN model at {model_path}")
            elif kaggle_path.exists():
                model_path = kaggle_path
                logger.info(f"Found Kaggle-trained CNN model at {model_path}")
            elif cnn_path.exists():
                model_path = cnn_path
                logger.info(f"Found local auto-trained CNN model at {model_path}")
            elif rf_path.exists():
                model_path = rf_path
                self.is_rf_model = True
                logger.info(f"Found robust Random Forest model at {model_path}")
            else:
                model_path = h5_path
        

        if not Path(model_path).exists():
            logger.warning(f"Mudra model not found at {model_path}. Using mock predictions.")
            self.model = None
        else:
            logger.info(f"Loading mudra model from {model_path}")
            try:
                if str(model_path).endswith('.pkl'):
                    import joblib
                    self.model = joblib.load(model_path)
                    self.is_rf_model = True
                else:
                    # CONFIG: Enable unsafe deserialization for trusted models to allow lambda/custom layers
                    try:
                        # Attempt Standard Load
                        logger.info("Attempting standard keras.models.load_model...")
                        # Add custom object scope if needed, but standard layers should be fine.
                        self.model = keras.models.load_model(str(model_path), compile=False)
                        
                        # Verify input shape
                        if hasattr(self.model, 'input_shape'):
                            shape = self.model.input_shape
                            if shape and len(shape) == 4:
                                self.input_shape = (shape[1], shape[2])
                                logger.info(f"Standard load success. Input shape: {self.input_shape}")
                        
                        # Re-derive class names if possible (rarely saved in model, so we might need mapping)
                        # We will assume mapping loaded below.
                        
                    except Exception as e_load:
                        logger.warning(f"Standard load failed: {e_load}. Falling back to reconstruction.")
                        
                        # Safe loading via reconstruction to avoid config issues
                        self.model, self.class_names_47 = self.reconstruct_and_load_model(model_path)
                        
                        # Auto-detect input shape
                        if hasattr(self.model, 'input_shape'):
                             # shape is usually (None, H, W, C)
                             shape = self.model.input_shape
                             if shape and len(shape) == 4:
                                 self.input_shape = (shape[1], shape[2])
                                 logger.info(f"Model input shape detected: {self.input_shape}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                logger.error(f"Failed to load model: {e}")
                self.model = None
        
        # Load class mappings if not established by reconstruction
        if hasattr(self, 'class_names_47'):
             # Use the specific mapping for this model
             self.num_classes = len(self.class_names_47)
             self.idx_to_class = {i: name for i, name in enumerate(self.class_names_47)}
             self.class_to_idx = {name: i for i, name in enumerate(self.class_names_47)}
             logger.info(f"Using robust 47-class mapping from reconstruction.")
        else:
            try:
                mapping = load_class_mappings()
                self.num_classes = mapping['num_classes']
                self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
                self.class_to_idx = mapping['class_to_idx']
            except FileNotFoundError:
                logger.warning("Class mapping not found. Using default mapping.")
                self.num_classes = 50
                self.idx_to_class = {i: f"Mudra_{i}" for i in range(self.num_classes)}
                self.class_to_idx = {v: k for k, v in self.idx_to_class.items()}
        
        # Feature extractor
        if extractor is None:
            self.extractor = FeatureExtractor()
        else:
            self.extractor = extractor
            
    def reconstruct_and_load_model(self, model_path):
        """
        Manually reconstruct EfficientNetB0 architecture and load weights 
        to bypass Keras config deserialization issues.
        """
        logger.info("Attempting robust model reconstruction...")
        from tensorflow.keras.applications import EfficientNetB0
        from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
        from tensorflow.keras.models import Model
        
        # Consistent 47 classes (All 50 from YAML minus Tripathaka which had 0 samples)
        # Assuming sorted order as per flow_from_dataframe
        yaml_names = ['Alapadmam', 'Anjali', 'Aralam', 'Ardhachandran', 'Ardhapathaka', 'Berunda', 'Bramaram', 
                      'Chakra', 'Chandrakala', 'Chaturam', 'Garuda', 'Hamsapaksha', 'Hamsasyam', 'Kangulam', 
                      'Kapith', 'Kapotham', 'Karkatta', 'Kartariswastika', 'Katakamukha', 'Katakavardhana', 
                      'Katrimukha', 'Khatva', 'Kilaka', 'Kurma', 'Matsya', 'Mayura', 'Mrigasirsha', 'Mukulam', 
                      'Mushti', 'Nagabandha', 'Padmakosha', 'Pasha', 'Pathaka', 'Pushpaputa', 'Sakata', 
                      'Samputa', 'Sarpasirsha', 'Shanka', 'Shivalinga', 'Shukatundam', 'Sikharam', 
                      'Simhamukham', 'Suchi', 'Swastikam', 'Tamarachudam', 'Tripathaka', 'Trishulam', 'Varaha']
                      
        # Exclude Tripathaka (and potentially verified empty ones, but Tripathaka is the known one)
        effective_classes = sorted([n for n in yaml_names if n != 'Tripathaka'])
        num_classes = len(effective_classes) # Should be 47
        
        base_model = EfficientNetB0(weights=None, include_top=False, input_shape=(224, 224, 3))
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        predictions = Dense(num_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=predictions)
        
        logger.info(f"Reconstructed architecture with {num_classes} output classes.")
        
        try:
             # Try topology-based loading first (failed earlier but worth trying with correct class count)
             model.load_weights(str(model_path))
             logger.info("Weights loaded successfully.")
        except ValueError as e:
             logger.warning(f"Standard loading failed: {e}. Trying by_name...")
             model.load_weights(str(model_path), by_name=True, skip_mismatch=True)
        
        return model, effective_classes
    
    def preprocess_landmarks(self, landmarks):
        """Normalize landmarks (same as training)."""
        if landmarks is None:
            return None
        if np.all(landmarks == 0):
            return landmarks
        
        normalized = landmarks.copy()
        for hand_idx in range(2):
            offset = hand_idx * 63
            hand_landmarks = landmarks[offset:offset + 63]
            if np.all(hand_landmarks == 0): continue
            
            wrist_x, wrist_y, wrist_z = hand_landmarks[0], hand_landmarks[1], hand_landmarks[2]
            for i in range(21):
                normalized[offset + i*3] -= wrist_x
                normalized[offset + i*3 + 1] -= wrist_y
                normalized[offset + i*3 + 2] -= wrist_z
            
            hand_span = np.max(np.abs(normalized[offset:offset + 63]))
            if hand_span > 0:
                normalized[offset:offset + 63] /= hand_span
        return normalized
    
    def predict_mudra(self, hand_landmarks, image=None, top_k=3):
        # CNN Inference (Auto-detected 32x32 or 224x224 input)
        if hasattr(self.model, 'input_shape'):
            if image is None: 
                logger.warning("Predict Mudra: Image is None!")
                return [("Unknown", 0.0)]
            
            target_h, target_w = self.input_shape
            
            # --- Hand Cropping Logic ---
            crop_success = False
            hand_crop = image
            
            try:
                h, w = image.shape[:2]
                landmarks_all = hand_landmarks.reshape(-1, 3) # (42, 3)
                valid_points = []
                for pt in landmarks_all:
                     # Check if point is valid (not all zeros)
                     if np.all(pt == 0): continue
                     # Denormalize
                     px = int(pt[0] * w)
                     py = int(pt[1] * h)
                     valid_points.append([px, py])
                
                if valid_points:
                    valid_points = np.array(valid_points)
                    x_min = np.min(valid_points[:, 0])
                    y_min = np.min(valid_points[:, 1])
                    x_max = np.max(valid_points[:, 0])
                    y_max = np.max(valid_points[:, 1])
                    
                    # Padding
                    pad = 40 # Increased padding for context
                    x_min = max(0, x_min - pad)
                    y_min = max(0, y_min - pad)
                    x_max = min(w, x_max + pad)
                    y_max = min(h, y_max + pad)
                    
                    if x_max > x_min and y_max > y_min:
                        hand_crop = image[y_min:y_max, x_min:x_max]
                        crop_success = True
                        # logger.info(f"Cropped hand: {hand_crop.shape}")
            except Exception as e:
                logger.warning(f"Cropping failed: {e}")
            
            # If no crop (no hands), we might want to return Unknown?
            # Or try full frame (which leads to Anjali)?
            # Better to return Unknown if we rely on landmarks.
            if not crop_success:
                 # If landmarks were all zero, we fall here.
                 # If we return Unknown, we avoid Anjali spam.
                 if np.all(hand_landmarks == 0):
                      return [("Unknown", 0.0)]
            
            # Ensure RGB (cv2 loads BGR)
            if len(hand_crop.shape) == 3 and hand_crop.shape[2] == 3:
                img_rgb = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = hand_crop
            
            # --- PAD TO SQUARE ---
            h, w = img_rgb.shape[:2]
            if h != w:
                max_dim = max(h, w)
                delta_w = max_dim - w
                delta_h = max_dim - h
                top, bottom = delta_h // 2, delta_h - (delta_h // 2)
                left, right = delta_w // 2, delta_w - (delta_w // 2)
                
                img_rgb = cv2.copyMakeBorder(
                    img_rgb, top, bottom, left, right, 
                    cv2.BORDER_CONSTANT, value=[0, 0, 0]
                )
                # logger.info(f"Padded crop from {w}x{h} to {max_dim}x{max_dim}")
                
            img_resized = cv2.resize(img_rgb, (target_w, target_h))
            img_norm = img_resized.astype(np.float32) / 255.0 # Normalize 0-1
            input_data = np.expand_dims(img_norm, axis=0)
            
            try:
                predictions = self.model.predict(input_data, verbose=0)[0]
                # logger.info(f"Predictions max: {np.max(predictions)}, argmax: {np.argmax(predictions)}")
            
                top_indices = np.argsort(predictions)[-top_k:][::-1]
                return [(self.idx_to_class.get(i, f"Unknown_{i}"), float(predictions[i])) for i in top_indices]
            except Exception as e:
                with open("cnn_inference_error.log", "a") as f:
                    import traceback
                    f.write(f"Prediction Error: {e}\n")
                    traceback.print_exc(file=f)
                logger.error(f"Prediction error: {e}")
                return []
        else:
             # Debug why hasattr failed
             try:
                 logger.warning(f"CNN skipped. model type: {type(self.model)}")
                 logger.warning(f"Has input_shape? {hasattr(self.model, 'input_shape')}")
             except: pass

        # Legacy/Hybrid Inference (Fall-through)
        landmarks_norm = self.preprocess_landmarks(hand_landmarks)
        if landmarks_norm is None:
             return [("Unknown", 0.0)]
             
        if self.model_type == 'hybrid':
             image_resized = cv2.resize(image, config.MUDRA_IMAGE_SIZE) / 255.0
             input_data = {'landmarks': np.expand_dims(landmarks_norm, axis=0), 'images': np.expand_dims(image_resized, axis=0)}
        elif self.model_type == 'landmark':
             input_data = np.expand_dims(landmarks_norm, axis=0)
        
        predictions = self.model.predict(input_data, verbose=0)[0]
        top_indices = np.argsort(predictions)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            class_name = self.idx_to_class.get(idx, f"Unknown_{idx}")
            confidence = float(predictions[idx])
            results.append((class_name, confidence))
        
        return results
    
    def predict_from_frame(self, frame, top_k=3):
        """
        Extract landmarks from frame and predict mudra.
        
        Args:
            frame: OpenCV BGR image
            top_k: Return top-k predictions
        
        Returns:
            List of (class_name, confidence) tuples
        """
        # Convert to RGB
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Extract hand landmarks
        hand_landmarks, _, _ = self.extractor.extract_landmarks(image_rgb)
        
        # Predict
        return self.predict_mudra(hand_landmarks, frame, top_k=top_k)
    
    def close(self):
        """Clean up resources."""
        self.extractor.close()


def test_predictor():
    """Test mudra predictor."""
    logging.basicConfig(level=logging.INFO, format=config.LOG_FORMAT)
    
    predictor = MudraPredictor(model_type='landmark')
    
    # Test with dummy data
    dummy_landmarks = np.random.randn(126)
    dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    results = predictor.predict_mudra(dummy_landmarks, dummy_image)
    
    logger.info("Test Predictions:")
    for class_name, confidence in results:
        logger.info(f"  {class_name}: {confidence:.4f}")
    
    predictor.close()


if __name__ == "__main__":
    test_predictor()

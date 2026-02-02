# ==========================================
# Mudra Analysis - Kaggle Training Script (Robust v2.1)
# ==========================================

# --- FIX: Protocol Buffers Version Mismatch ---
import os
print("Installing compatible protobuf...")
os.system("pip install protobuf==3.20.3")
print("Protobuf installed. Proceeding...")
# ----------------------------------------------

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# --- Config ---
def find_dataset_dirs():
    print("Searching for dataset...")
    search_path = '/kaggle/input'
    
    # Heuristic: Look for 'train' directory
    train_dir = None
    val_dir = None
    
    for root, dirs, files in os.walk(search_path):
        if 'train' in dirs:
            possible_train = os.path.join(root, 'train')
            # Check if it has subdirectories (classes)
            if any(os.path.isdir(os.path.join(possible_train, d)) for d in os.listdir(possible_train)):
                train_dir = possible_train
                # Assume val is sibling
                if 'val' in dirs:
                    val_dir = os.path.join(root, 'val')
                break
                
    if not train_dir:
        print("Could not auto-detect train directory. Listing /kaggle/input:")
        os.system("ls -R /kaggle/input | head -n 20")
        raise FileNotFoundError("Training directory not found.")
        
    if not val_dir:
         print("Warning: Validation directory not found. Will use split if needed (not implemented here).")

    return train_dir, val_dir

# Robust Data Handling (Copy to Writeable Directory)
import shutil

def setup_writable_data(train_source, val_source):
    working_dir = '/kaggle/working/data'
    if os.path.exists(working_dir):
        print("Cleaning up previous run...")
        shutil.rmtree(working_dir)
    
    os.makedirs(working_dir)
    
    print(f"Copying Train Data to {working_dir} (This may take a few minutes)...")
    shutil.copytree(train_source, os.path.join(working_dir, 'train'))
    
    print(f"Copying Val Data to {working_dir}...")
    shutil.copytree(val_source, os.path.join(working_dir, 'val'))
    
    return os.path.join(working_dir, 'train'), os.path.join(working_dir, 'val')

print("Setting up writable dataset...")
TRAIN_DIR_SRC, VAL_DIR_SRC = find_dataset_dirs()
TRAIN_DIR, VAL_DIR = setup_writable_data(TRAIN_DIR_SRC, VAL_DIR_SRC)

print(f"Writable Training Data: {TRAIN_DIR}")
print(f"Writable Validation Data: {VAL_DIR}")

# Params
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-3

# Robust Image Verification
from PIL import Image
def verify_images(directory):
    print(f"Verifying integrity of images in {directory}...")
    corrupted = 0
    valid = 0
    for root, dirs, files in os.walk(directory):
        for file in files:
            args = os.path.join(root, file)
            try:
                with Image.open(args) as img:
                    img.verify() 
                valid += 1
            except (IOError, SyntaxError) as e:
                print(f"Removing corrupted image: {args}")
                os.remove(args)
                corrupted += 1
    print(f"Verification Done. Valid: {valid}, Corrupted/Removed: {corrupted}")

print("Checking Train Data...")
verify_images(TRAIN_DIR)
print("Checking Val Data...")
verify_images(VAL_DIR)

# Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

val_datagen = ImageDataGenerator(rescale=1./255)

print("Loading Train Generator...")
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=True
)

print("Loading Val Generator...")
val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

NUM_CLASSES = len(train_generator.class_indices)
print(f"Detected {NUM_CLASSES} classes.")

# Save class indices for inference later
class_indices = train_generator.class_indices
# Invert: index -> name
idx_to_class = {v: k for k, v in class_indices.items()}
print("Class Indices (Sample):", list(idx_to_class.items())[:5])

# Calculate Class Weights (Optional/Advanced - Simple version here)
# To handle remaining imbalance if any (though we balanced it)
from sklearn.utils import class_weight
train_classes = train_generator.classes
class_weights = class_weight.compute_class_weight(
    class_weight='balanced',
    classes=np.unique(train_classes),
    y=train_classes
)
class_weights_dict = dict(enumerate(class_weights))
print("Class weights calculated.")

# Model
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Callbacks
save_path = '/kaggle/working/mudra_cnn_model_kaggle.h5'
callbacks = [
    ModelCheckpoint(save_path, monitor='val_accuracy', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1),
    EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True)
]

print("Starting training...")
history = model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    class_weight=class_weights_dict
)

print(f"Training Done. Model saved to {save_path}")
print("You can download the model from the 'Output' section of the notebook.")

# Save class map
import json
with open('/kaggle/working/class_indices.json', 'w') as f:
    json.dump(class_indices, f)
print("Saved class_indices.json")


import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import json
import yaml

from src import config

# Configuration overrides
AUG_TRAIN_DIR = r"c:\aashish programming files\freelance\nithya-analysis ml model\data\mudras\kaggle_50_mudras\images\train_augmented"
VAL_DIR = r"c:\aashish programming files\freelance\nithya-analysis ml model\data\mudras\kaggle_50_mudras\images\val"
YAML_PATH = r"c:\aashish programming files\freelance\nithya-analysis ml model\data\mudras\kaggle_50_mudras\dataset_augmented.yaml"

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 20 # Can increase
LEARNING_RATE = 1e-3

def load_class_names():
    with open(YAML_PATH, 'r') as f:
        data = yaml.safe_load(f)
    return data['names']

def get_class_from_filename(filename, class_names):
    # Sort by length to match longest first
    sorted_names = sorted(class_names, key=len, reverse=True)
    for name in sorted_names:
        if filename.startswith(name):
            return name
    return None

def create_dataframe(directory, class_names):
    print(f"Scanning {directory}...")
    files = glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.png"))
    data = []
    
    for f in files:
        filename = os.path.basename(f)
        label = get_class_from_filename(filename, class_names)
        if label:
            data.append({'filename': f, 'class': label})
            
    df = pd.DataFrame(data)
    print(f"Found {len(df)} images.")
    if not df.empty:
        print(f"Class counts: {df['class'].value_counts().head()}")
    return df

def main():
    # 0. Setup
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        print(f"Using GPU: {gpus}")
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as e:
            print(e)
    else:
        print("Using CPU")

    class_names = load_class_names()
    print(f"Classes to train: {len(class_names)}")
    
    # 1. Prepare DataFrames
    train_df = create_dataframe(AUG_TRAIN_DIR, class_names)
    val_df = create_dataframe(VAL_DIR, class_names)
    
    if train_df.empty or val_df.empty:
        print("Error: No data found.")
        return

    # 2. Data Generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_dataframe(
        val_df,
        x_col='filename',
        y_col='class',
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    # Save Class Mapping immediately
    mapping = {
        'class_to_idx': train_generator.class_indices,
        'idx_to_class': {v: k for k, v in train_generator.class_indices.items()},
        'num_classes': len(train_generator.class_indices)
    }
    mapping_path = os.path.join(config.PROCESSED_MUDRA_FEATURES, 'class_mapping.json')
    os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
    with open(mapping_path, 'w') as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved class mapping to {mapping_path}")

    # 3. Model
    base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=IMG_SIZE + (3,))
    
    # Freeze base model first? Or train all?
    # Unfreeze top few layers usually better.
    base_model.trainable = True # Fine-tune all for best capacity given 1500 images/class (70k total)
    
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.2)(x)
    predictions = Dense(len(class_names), activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    
    # 4. Train
    checkpoint_path = os.path.join(config.MODEL_DIR, 'mudra_cnn_model.h5')
    
    callbacks = [
        ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_lr=1e-6),
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1)
    ]
    
    print(f"Starting training on {len(train_df)} images for {EPOCHS} epochs...")
    
    history = model.fit(
        train_generator,
        epochs=EPOCHS,
        validation_data=val_generator,
        callbacks=callbacks
    )
    
    print("Training complete.")

if __name__ == "__main__":
    main()

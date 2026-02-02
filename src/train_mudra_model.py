"""
Training Script for Mudra Classification Model
Implements state-of-the-art training with advanced techniques.
"""
import os
import sys
import logging
import argparse
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import callbacks
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

from src import config
from src.models.mudra_classifier import create_mudra_model
from src.dataset_loader import load_class_mappings

logger = logging.getLogger(__name__)


class MudraTrainer:
    """Trainer class for mudra classification."""
    
    def __init__(self, model_type='hybrid', num_classes=None, use_pretrained=True):
        self.model_type = model_type
        self.use_pretrained = use_pretrained
        
        # Load class mappings
        try:
            mapping = load_class_mappings()
            self.num_classes = mapping['num_classes']
            self.idx_to_class = {int(k): v for k, v in mapping['idx_to_class'].items()}
        except FileNotFoundError:
            logger.warning("Class mapping not found. Using default num_classes.")
            self.num_classes = num_classes or 50
            self.idx_to_class = {i: f"Class_{i}" for i in range(self.num_classes)}
        
        self.model = None
        self.history = None
        
    def load_data(self):
        """Load processed datasets."""
        logger.info("Loading processed datasets...")
        
        data_dir = Path(config.PROCESSED_MUDRA_FEATURES)
        
        # Load training data
        X_landmarks_train = np.load(data_dir / 'X_landmarks_train.npy')
        X_images_train = np.load(data_dir / 'X_images_train.npy')
        y_train = np.load(data_dir / 'y_labels_train.npy')
        
        # Load validation data
        X_landmarks_val = np.load(data_dir / 'X_landmarks_val.npy')
        X_images_val = np.load(data_dir / 'X_images_val.npy')
        y_val = np.load(data_dir / 'y_labels_val.npy')
        
        # Load test data
        X_landmarks_test = np.load(data_dir / 'X_landmarks_test.npy')
        X_images_test = np.load(data_dir / 'X_images_test.npy')
        y_test = np.load(data_dir / 'y_labels_test.npy')
        
        logger.info(f"Training samples: {len(y_train)}")
        logger.info(f"Validation samples: {len(y_val)}")
        logger.info(f"Test samples: {len(y_test)}")
        
        # Prepare data based on model type
        if self.model_type == 'hybrid':
            self.train_data = ({'landmarks': X_landmarks_train, 'images': X_images_train}, y_train)
            self.val_data = ({'landmarks': X_landmarks_val, 'images': X_images_val}, y_val)
            self.test_data = ({'landmarks': X_landmarks_test, 'images': X_images_test}, y_test)
        elif self.model_type == 'landmark':
            self.train_data = (X_landmarks_train, y_train)
            self.val_data = (X_landmarks_val, y_val)
            self.test_data = (X_landmarks_test, y_test)
        elif self.model_type == 'image':
            self.train_data = (X_images_train, y_train)
            self.val_data = (X_images_val, y_val)
            self.test_data = (X_images_test, y_test)
        
        return self.train_data, self.val_data, self.test_data
    
    def build_model(self):
        """Build and compile model."""
        logger.info(f"Building {self.model_type} model with {self.num_classes} classes...")
        
        self.model = create_mudra_model(
            model_type=self.model_type,
            num_classes=self.num_classes,
            use_pretrained=self.use_pretrained
        )
        
        # Compile with advanced optimizer and metrics
        optimizer = Adam(learning_rate=config.LEARNING_RATE)
        
        self.model.compile(
            optimizer=optimizer,
            loss='sparse_categorical_crossentropy',
            metrics=[
                'accuracy',
                keras.metrics.SparseTopKCategoricalAccuracy(k=3, name='top_3_accuracy'),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top_5_accuracy')
            ]
        )
        
        logger.info("Model compiled successfully!")
        return self.model
    
    def get_callbacks(self):
        """Get training callbacks."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Model checkpoint
        checkpoint_path = Path(config.MODEL_DIR) / f'mudra_{self.model_type}_{timestamp}_best.h5'
        checkpoint = callbacks.ModelCheckpoint(
            checkpoint_path,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
        
        # Early stopping
        early_stop = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=config.EARLY_STOPPING_PATIENCE,
            restore_best_weights=True,
            verbose=1
        )
        
        # Learning rate reduction
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=config.REDUCE_LR_PATIENCE,
            min_lr=1e-7,
            verbose=1
        )
        
        # TensorBoard
        log_dir = Path(config.MODEL_DIR) / 'logs' / f'{self.model_type}_{timestamp}'
        tensorboard = callbacks.TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        
        # CSV Logger
        csv_path = Path(config.MODEL_DIR) / f'training_{self.model_type}_{timestamp}.csv'
        csv_logger = callbacks.CSVLogger(csv_path)
        
        return [checkpoint, early_stop, reduce_lr, tensorboard, csv_logger]
    
    def train(self, epochs=None):
        """Train the model."""
        if self.model is None:
            self.build_model()
        
        epochs = epochs or config.EPOCHS
        
        logger.info(f"Starting training for {epochs} epochs...")
        
        X_train, y_train = self.train_data
        X_val, y_val = self.val_data
        
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=config.BATCH_SIZE,
            callbacks=self.get_callbacks(),
            verbose=1
        )
        
        logger.info("Training complete!")
        return self.history
    
    def evaluate(self):
        """Evaluate model on test set."""
        logger.info("Evaluating on test set...")
        
        X_test, y_test = self.test_data
        
        results = self.model.evaluate(X_test, y_test, verbose=1)
        
        metrics = dict(zip(self.model.metrics_names, results))
        
        logger.info("Test Results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def predict_and_analyze(self):
        """Generate predictions and analysis."""
        logger.info("Generating predictions...")
        
        X_test, y_test = self.test_data
        
        # Predict
        y_pred_probs = self.model.predict(X_test)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        # Classification report
        class_names = [self.idx_to_class[i] for i in range(self.num_classes)]
        report = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)
        
        logger.info("\nClassification Report:")
        print(report)
        
        # Save report
        report_path = Path(config.MODEL_DIR) / f'classification_report_{self.model_type}.txt'
        with open(report_path, 'w') as f:
            f.write(report)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        return y_pred, cm, report
    
    def plot_training_history(self):
        """Plot training history."""
        if self.history is None:
            logger.warning("No training history to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Accuracy
        axes[0, 0].plot(self.history.history['accuracy'], label='Train')
        axes[0, 0].plot(self.history.history['val_accuracy'], label='Val')
        axes[0, 0].set_title('Model Accuracy')
        axes[0, 0].set_ylabel('Accuracy')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Loss
        axes[0, 1].plot(self.history.history['loss'], label='Train')
        axes[0, 1].plot(self.history.history['val_loss'], label='Val')
        axes[0, 1].set_title('Model Loss')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Top-3 Accuracy
        axes[1, 0].plot(self.history.history['top_3_accuracy'], label='Train')
        axes[1, 0].plot(self.history.history['val_top_3_accuracy'], label='Val')
        axes[1, 0].set_title('Top-3 Accuracy')
        axes[1, 0].set_ylabel('Accuracy')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Top-5 Accuracy
        axes[1, 1].plot(self.history.history['top_5_accuracy'], label='Train')
        axes[1, 1].plot(self.history.history['val_top_5_accuracy'], label='Val')
        axes[1, 1].set_title('Top-5 Accuracy')
        axes[1, 1].set_ylabel('Accuracy')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        
        # Save
        save_path = Path(config.MODEL_DIR) / f'training_history_{self.model_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Training history saved to {save_path}")
        
        plt.close()
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix."""
        plt.figure(figsize=(20, 16))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized,
            annot=False,
            fmt='.2f',
            cmap='Blues',
            xticklabels=[self.idx_to_class[i] for i in range(self.num_classes)],
            yticklabels=[self.idx_to_class[i] for i in range(self.num_classes)],
            cbar_kws={'label': 'Normalized Frequency'}
        )
        
        plt.title(f'Confusion Matrix - {self.model_type.upper()} Model')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        
        # Save
        save_path = Path(config.MODEL_DIR) / f'confusion_matrix_{self.model_type}.png'
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Confusion matrix saved to {save_path}")
        
        plt.close()
    
    def save_model(self, filename=None):
        """Save trained model."""
        if filename is None:
            filename = f'mudra_classifier_{self.model_type}_final.h5'
        
        save_path = Path(config.MODEL_DIR) / filename
        self.model.save(save_path)
        logger.info(f"Model saved to {save_path}")
        
        return save_path


def main():
    parser = argparse.ArgumentParser(description='Train Mudra Classification Model')
    parser.add_argument('--model_type', type=str, default='hybrid',
                       choices=['hybrid', 'landmark', 'image'],
                       help='Model architecture type')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--no_pretrained', action='store_true',
                       help='Do not use pretrained ImageNet weights')
    parser.add_argument('--evaluate_only', action='store_true',
                       help='Only evaluate existing model')
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT,
        handlers=[
            logging.FileHandler(Path(config.MODEL_DIR) / f'training_{args.model_type}.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Create trainer
    trainer = MudraTrainer(
        model_type=args.model_type,
        use_pretrained=not args.no_pretrained
    )
    
    # Load data
    try:
        trainer.load_data()
    except FileNotFoundError as e:
        logger.error(f"Data files not found: {e}")
        logger.error("Please run mudra processing first: python src/mudra_processor.py")
        return 1
    
    if not args.evaluate_only:
        # Build and train
        trainer.build_model()
        trainer.train(epochs=args.epochs)
        
        # Plot training history
        trainer.plot_training_history()
        
        # Save model
        trainer.save_model()
    else:
        # Load existing model
        model_path = Path(config.MODEL_DIR) / f'mudra_classifier_{args.model_type}_final.h5'
        if not model_path.exists():
            logger.error(f"Model not found: {model_path}")
            return 1
        
        trainer.model = keras.models.load_model(model_path)
        logger.info(f"Loaded model from {model_path}")
    
    # Evaluate
    metrics = trainer.evaluate()
    
    # Detailed analysis
    y_pred, cm, report = trainer.predict_and_analyze()
    
    # Plot confusion matrix
    trainer.plot_confusion_matrix(cm)
    
    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Model Type: {args.model_type.upper()}")
    logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Top-3 Accuracy: {metrics['top_3_accuracy']:.4f}")
    logger.info(f"Top-5 Accuracy: {metrics['top_5_accuracy']:.4f}")
    logger.info("="*60)
    
    return 0


if __name__ == "__main__":
    exit(main())

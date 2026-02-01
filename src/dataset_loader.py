"""
Dataset Loader for Mudra Classification
Handles loading and organizing multiple mudra datasets with consistent interface.
"""
import os
import logging
import json
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.model_selection import train_test_split
import cv2
from src import config

logger = logging.getLogger(__name__)

class MudraDatasetLoader:
    """Unified loader for all mudra datasets."""
    
    def __init__(self, include_kaggle=True, include_asamyuktha=True):
        self.include_kaggle = include_kaggle
        self.include_asamyuktha = include_asamyuktha
        self.class_to_idx = {}
        self.idx_to_class = {}
        self.data_samples = []  # List of (image_path, class_idx, dataset_source)
        
    def load_kaggle_dataset(self):
        """Load Kaggle 50 mudras dataset."""
        logger.info("Loading Kaggle dataset...")
        kaggle_dir = Path(config.RAW_MUDRAS_KAGGLE_DIR)
        
        if not kaggle_dir.exists():
            logger.warning(f"Kaggle dataset not found at {kaggle_dir}")
            return 0
        
        count = 0
        for class_dir in sorted(kaggle_dir.iterdir()):
            if class_dir.is_dir():
                class_name = class_dir.name
                
                # Add to class mapping if not present
                if class_name not in self.class_to_idx:
                    idx = len(self.class_to_idx)
                    self.class_to_idx[class_name] = idx
                    self.idx_to_class[idx] = class_name
                
                class_idx = self.class_to_idx[class_name]
                
                # Load all images from this class
                for img_file in class_dir.glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        self.data_samples.append({
                            'path': str(img_file),
                            'class_idx': class_idx,
                            'class_name': class_name,
                            'source': 'kaggle'
                        })
                        count += 1
        
        logger.info(f"Loaded {count} images from Kaggle dataset")
        return count
    
    def load_asamyuktha_dataset(self):
        """Load Asamyuktha 27 mudras dataset."""
        logger.info("Loading Asamyuktha dataset...")
        asamyuktha_dir = Path(config.RAW_MUDRAS_ASAMYUKTHA_DIR)
        
        if not asamyuktha_dir.exists():
            logger.warning(f"Asamyuktha dataset not found at {asamyuktha_dir}")
            return 0
        
        count = 0
        # Load from train/test/val splits
        for split in ['train', 'test', 'val']:
            split_dir = asamyuktha_dir / split
            if split_dir.exists():
                for class_dir in sorted(split_dir.iterdir()):
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        
                        # Add to class mapping if not present
                        if class_name not in self.class_to_idx:
                            idx = len(self.class_to_idx)
                            self.class_to_idx[class_name] = idx
                            self.idx_to_class[idx] = class_name
                        
                        class_idx = self.class_to_idx[class_name]
                        
                        # Load all images from this class
                        for img_file in class_dir.glob('*.*'):
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                self.data_samples.append({
                                    'path': str(img_file),
                                    'class_idx': class_idx,
                                    'class_name': class_name,
                                    'source': 'asamyuktha',
                                    'split': split
                                })
                                count += 1
        
        logger.info(f"Loaded {count} images from Asamyuktha dataset")
        return count
    
    def load_all(self):
        """Load all enabled datasets."""
        total = 0
        
        if self.include_kaggle:
            total += self.load_kaggle_dataset()
        
        if self.include_asamyuktha:
            total += self.load_asamyuktha_dataset()
        
        logger.info(f"Total: {total} images across {len(self.class_to_idx)} classes")
        
        # Save class mappings
        self.save_class_mappings()
        
        return total
    
    def save_class_mappings(self):
        """Save class to index mappings."""
        mapping_path = Path(config.PROCESSED_MUDRA_FEATURES) / 'class_mapping.json'
        mapping_path.parent.mkdir(parents=True, exist_ok=True)
        
        mapping = {
            'class_to_idx': self.class_to_idx,
            'idx_to_class': {int(k): v for k, v in self.idx_to_class.items()},
            'num_classes': len(self.class_to_idx)
        }
        
        with open(mapping_path, 'w') as f:
            json.dump(mapping, f, indent=2)
        
        logger.info(f"Saved class mappings to {mapping_path}")
    
    def create_splits(self, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, random_state=42):
        """
        Create train/val/test splits.
        For Asamyuktha, respect existing splits.
        For Kaggle, create new splits.
        """
        # Separate by source
        kaggle_samples = [s for s in self.data_samples if s['source'] == 'kaggle']
        asamyuktha_samples = [s for s in self.data_samples if s['source'] == 'asamyuktha']
        
        train_samples = []
        val_samples = []
        test_samples = []
        
        # Handle Asamyuktha (already has splits)
        for sample in asamyuktha_samples:
            if sample['split'] == 'train':
                train_samples.append(sample)
            elif sample['split'] == 'val':
                val_samples.append(sample)
            elif sample['split'] == 'test':
                test_samples.append(sample)
        
        # Handle Kaggle (create splits)
        if kaggle_samples:
            # Group by class for stratified split
            class_groups = defaultdict(list)
            for sample in kaggle_samples:
                class_groups[sample['class_idx']].append(sample)
            
            for class_idx, samples in class_groups.items():
                # First split: train vs (val + test)
                train_cls, temp_cls = train_test_split(
                    samples,
                    train_size=train_ratio,
                    random_state=random_state
                )
                
                # Second split: val vs test
                val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
                val_cls, test_cls = train_test_split(
                    temp_cls,
                    train_size=val_ratio_adjusted,
                    random_state=random_state
                )
                
                train_samples.extend(train_cls)
                val_samples.extend(val_cls)
                test_samples.extend(test_cls)
        
        logger.info(f"Split: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
        
        return train_samples, val_samples, test_samples
    
    def get_class_distribution(self):
        """Get distribution of samples per class."""
        distribution = defaultdict(int)
        for sample in self.data_samples:
            distribution[sample['class_name']] += 1
        return distribution
    
    def load_image(self, image_path, target_size=(224, 224)):
        """Load and preprocess an image."""
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, target_size)
        return img
    
    def get_statistics(self):
        """Get dataset statistics."""
        dist = self.get_class_distribution()
        
        stats = {
            'total_samples': len(self.data_samples),
            'num_classes': len(self.class_to_idx),
            'samples_per_class': {
                'min': min(dist.values()) if dist else 0,
                'max': max(dist.values()) if dist else 0,
                'mean': np.mean(list(dist.values())) if dist else 0,
                'std': np.std(list(dist.values())) if dist else 0
            },
            'class_distribution': dict(dist)
        }
        
        return stats


def load_class_mappings():
    """Load saved class mappings."""
    mapping_path = Path(config.PROCESSED_MUDRA_FEATURES) / 'class_mapping.json'
    
    if not mapping_path.exists():
        raise FileNotFoundError(f"Class mapping not found at {mapping_path}. Run dataset loading first.")
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    return mapping


def main():
    """Test dataset loading."""
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT
    )
    
    loader = MudraDatasetLoader(include_kaggle=True, include_asamyuktha=True)
    total = loader.load_all()
    
    if total == 0:
        logger.error("No datasets found! Please download datasets first.")
        return 1
    
    # Print statistics
    stats = loader.get_statistics()
    print(f"\nDataset Statistics:")
    print(f"  Total Samples: {stats['total_samples']}")
    print(f"  Number of Classes: {stats['num_classes']}")
    print(f"  Samples per Class:")
    print(f"    Min: {stats['samples_per_class']['min']}")
    print(f"    Max: {stats['samples_per_class']['max']}")
    print(f"    Mean: {stats['samples_per_class']['mean']:.1f}")
    print(f"    Std: {stats['samples_per_class']['std']:.1f}")
    
    # Create splits
    train, val, test = loader.create_splits()
    print(f"\nSplits: Train={len(train)}, Val={len(val)}, Test={len(test)}")
    
    return 0


if __name__ == "__main__":
    exit(main())

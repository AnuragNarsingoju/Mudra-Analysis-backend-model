"""
Dataset Verification Utility
Verifies downloaded mudra datasets and reports statistics.
"""
import os
import logging
from pathlib import Path
from collections import defaultdict
import cv2
from src import config

logger = logging.getLogger(__name__)

class DatasetVerifier:
    def __init__(self):
        self.stats = {
            'kaggle': defaultdict(int),
            'asamyuktha': defaultdict(int),
            'total_classes': set(),
            'corrupted_files': []
        }
    
    def verify_image(self, image_path):
        """Check if image can be loaded."""
        try:
            img = cv2.imread(str(image_path))
            if img is None:
                return False
            return True
        except Exception as e:
            logger.error(f"Error loading {image_path}: {e}")
            return False
    
    def verify_kaggle_dataset(self):
        """Verify Kaggle 50 mudras dataset."""
        logger.info("Verifying Kaggle dataset...")
        kaggle_dir = Path(config.RAW_MUDRAS_KAGGLE_DIR)
        
        if not kaggle_dir.exists():
            logger.warning(f"Kaggle dataset not found at {kaggle_dir}")
            logger.info("Please download from: https://www.kaggle.com/datasets/krithi9977/bharatanatyam-mudra-dataset-balanced/data")
            return False
        
        # Expected structure: kaggle_50_mudras/ClassName/*.jpg
        for class_dir in kaggle_dir.iterdir():
            if class_dir.is_dir():
                class_name = class_dir.name
                self.stats['total_classes'].add(class_name)
                
                for img_file in class_dir.glob('*.*'):
                    if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                        if self.verify_image(img_file):
                            self.stats['kaggle'][class_name] += 1
                        else:
                            self.stats['corrupted_files'].append(str(img_file))
        
        logger.info(f"Kaggle dataset: {len(self.stats['kaggle'])} classes found")
        return True
    
    def verify_asamyuktha_dataset(self):
        """Verify Asamyuktha 27 mudras dataset."""
        logger.info("Verifying Asamyuktha dataset...")
        asamyuktha_dir = Path(config.RAW_MUDRAS_ASAMYUKTHA_DIR)
        
        if not asamyuktha_dir.exists():
            logger.warning(f"Asamyuktha dataset not found at {asamyuktha_dir}")
            logger.info("Please download from GitHub repo: rohitreddy21122000/Asamyuktha-Mudras-classification")
            return False
        
        # Check train/test/val splits
        for split in ['train', 'test', 'val']:
            split_dir = asamyuktha_dir / split
            if split_dir.exists():
                for class_dir in split_dir.iterdir():
                    if class_dir.is_dir():
                        class_name = class_dir.name
                        self.stats['total_classes'].add(class_name)
                        
                        for img_file in class_dir.glob('*.*'):
                            if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
                                if self.verify_image(img_file):
                                    self.stats['asamyuktha'][class_name] += 1
                                else:
                                    self.stats['corrupted_files'].append(str(img_file))
        
        logger.info(f"Asamyuktha dataset: {len(self.stats['asamyuktha'])} classes found")
        return True
    
    def print_report(self):
        """Print comprehensive dataset report."""
        print("\n" + "="*60)
        print("DATASET VERIFICATION REPORT")
        print("="*60 + "\n")
        
        # Kaggle Dataset
        if self.stats['kaggle']:
            print("KAGGLE DATASET (50 Mudras)")
            print("-" * 60)
            total_kaggle = sum(self.stats['kaggle'].values())
            print(f"Total Classes: {len(self.stats['kaggle'])}")
            print(f"Total Images: {total_kaggle}")
            print(f"Average per Class: {total_kaggle / len(self.stats['kaggle']):.1f}")
            
            # Show class distribution
            print("\nClass Distribution:")
            for class_name, count in sorted(self.stats['kaggle'].items()):
                print(f"  {class_name:30s} : {count:4d} images")
            print()
        else:
            print("⚠️  KAGGLE DATASET NOT FOUND")
            print("   Download from: https://www.kaggle.com/datasets/krithi9977/bharatanatyam-mudra-dataset-balanced/data")
            print(f"   Place in: {config.RAW_MUDRAS_KAGGLE_DIR}\n")
        
        # Asamyuktha Dataset
        if self.stats['asamyuktha']:
            print("\nASAMYUKTHA DATASET (27 Single-Hand Mudras)")
            print("-" * 60)
            total_asamyuktha = sum(self.stats['asamyuktha'].values())
            print(f"Total Classes: {len(self.stats['asamyuktha'])}")
            print(f"Total Images: {total_asamyuktha}")
            print(f"Average per Class: {total_asamyuktha / len(self.stats['asamyuktha']):.1f}")
            
            # Show class distribution
            print("\nClass Distribution:")
            for class_name, count in sorted(self.stats['asamyuktha'].items()):
                print(f"  {class_name:30s} : {count:4d} images")
            print()
        else:
            print("⚠️  ASAMYUKTHA DATASET NOT FOUND")
            print("   Links:")
            print("   - Train: https://drive.google.com/drive/folders/1tbYvB251jMLqDNtcOiF9DWix_eV51HXd")
            print("   - Test: https://drive.google.com/drive/folders/1AWB7BPL_vW9Uqs9fuaT-CkBrQzAFp_j5")
            print("   - Val: https://drive.google.com/drive/folders/1dlZZSFhSz8O1Ibi9HphqbVs0-SzArykm")
            print(f"   Place in: {config.RAW_MUDRAS_ASAMYUKTHA_DIR}\n")
        
        # Overall Summary
        print("\nOVERALL SUMMARY")
        print("-" * 60)
        print(f"Total Unique Classes: {len(self.stats['total_classes'])}")
        print(f"Total Images: {sum(self.stats['kaggle'].values()) + sum(self.stats['asamyuktha'].values())}")
        
        # Corrupted files
        if self.stats['corrupted_files']:
            print(f"\n⚠️  Corrupted Files Found: {len(self.stats['corrupted_files'])}")
            for f in self.stats['corrupted_files'][:10]:
                print(f"  - {f}")
            if len(self.stats['corrupted_files']) > 10:
                print(f"  ... and {len(self.stats['corrupted_files']) - 10} more")
        else:
            print("\n✓ No corrupted files found")
        
        print("\n" + "="*60 + "\n")
        
        # Recommendations
        if not self.stats['kaggle'] and not self.stats['asamyuktha']:
            print("⚠️  ACTION REQUIRED: Please download at least one dataset to proceed")
        elif len(self.stats['total_classes']) < 20:
            print("⚠️  WARNING: Limited classes detected. Consider downloading both datasets.")
        else:
            print("✓ Datasets look good! Ready to proceed with processing.")
        
        print()
    
    def run(self):
        """Run full verification."""
        kaggle_ok = self.verify_kaggle_dataset()
        asamyuktha_ok = self.verify_asamyuktha_dataset()
        self.print_report()
        
        return kaggle_ok or asamyuktha_ok


def main():
    logging.basicConfig(
        level=logging.INFO,
        format=config.LOG_FORMAT
    )
    
    verifier = DatasetVerifier()
    success = verifier.run()
    
    if not success:
        logger.error("Dataset verification failed. Please download datasets.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

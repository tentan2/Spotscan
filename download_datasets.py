"""
Dataset Download Script
Downloads and sets up Food-101 and Nutrition5k datasets
"""

import os
import sys
import zipfile
import tarfile
import requests
from pathlib import Path
import kaggle
from tqdm import tqdm
import shutil

class DatasetDownloader:
    """Downloads and prepares food datasets"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        
    def download_food101(self):
        """Download Food-101 dataset from Kaggle"""
        print("Downloading Food-101 dataset...")
        
        # Food-101 dataset info
        dataset_name = "danbrice/food-101"
        extract_dir = self.data_dir / "food-101"
        
        try:
            # Download using kaggle API
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(extract_dir), 
                unzip=True
            )
            print(f"Food-101 dataset downloaded and extracted to {extract_dir}")
            
        except Exception as e:
            print(f"Error downloading Food-101: {e}")
            print("Please ensure you have configured Kaggle API credentials")
            return False
            
        return True
    
    def download_nutrition5k(self):
        """Download Nutrition5k dataset"""
        print("Downloading Nutrition5k dataset...")
        
        # Nutrition5k dataset info
        dataset_name = "cdg121/nutrition5k"
        extract_dir = self.data_dir / "nutrition5k"
        
        try:
            # Download using kaggle API
            kaggle.api.dataset_download_files(
                dataset_name, 
                path=str(extract_dir), 
                unzip=True
            )
            print(f"Nutrition5k dataset downloaded and extracted to {extract_dir}")
            
        except Exception as e:
            print(f"Error downloading Nutrition5k: {e}")
            print("Please ensure you have configured Kaggle API credentials")
            return False
            
        return True
    
    def verify_food101_structure(self):
        """Verify Food-101 dataset structure"""
        food101_dir = self.data_dir / "food-101"
        
        required_files = [
            "meta/meta/classes.txt",
            "meta/meta/train.txt",
            "meta/meta/test.txt"
        ]
        
        required_dirs = [
            "images"
        ]
        
        missing = []
        
        for file_path in required_files:
            if not (food101_dir / file_path).exists():
                missing.append(file_path)
        
        for dir_path in required_dirs:
            if not (food101_dir / dir_path).exists():
                missing.append(dir_path)
        
        if missing:
            print(f"Missing Food-101 files/directories: {missing}")
            return False
        
        print("Food-101 dataset structure verified")
        return True
    
    def verify_nutrition5k_structure(self):
        """Verify Nutrition5k dataset structure"""
        nutrition5k_dir = self.data_dir / "nutrition5k"
        
        # Check for typical Nutrition5k structure
        if nutrition5k_dir.exists():
            contents = list(nutrition5k_dir.iterdir())
            print(f"Nutrition5k directory contents: {[c.name for c in contents[:5]]}")
            return True
        else:
            print("Nutrition5k directory not found")
            return False
    
    def create_food101_metadata(self):
        """Create metadata files for Food-101"""
        food101_dir = self.data_dir / "food-101"
        
        # Read classes
        classes_file = food101_dir / "meta/meta/classes.txt"
        if classes_file.exists():
            with open(classes_file, 'r') as f:
                classes = [line.strip() for line in f.readlines()]
            
            # Create class mapping
            class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
            
            # Save metadata
            metadata = {
                'num_classes': len(classes),
                'classes': classes,
                'class_to_idx': class_to_idx
            }
            
            import json
            with open(food101_dir / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"Food-101 metadata created: {len(classes)} classes")
    
    def setup_all(self):
        """Download and setup all datasets"""
        print("Starting dataset download and setup...")
        
        # Download Food-101
        if self.download_food101():
            self.verify_food101_structure()
            self.create_food101_metadata()
        
        # Download Nutrition5k
        if self.download_nutrition5k():
            self.verify_nutrition5k_structure()
        
        print("Dataset setup complete!")

def main():
    """Main function to run dataset setup"""
    downloader = DatasetDownloader()
    downloader.setup_all()

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Cataract Dataset Preprocessing Pipeline

This script processes cataract datasets through three main steps:
1. Deduplication (remove duplicate images)
2. Cropping (crop images to focus on eye regions)
3. Train/Test Split (ensure left/right eyes from same person don't split)

Author: AI Assistant
Date: 2025-02-10
"""

import os
import shutil
import hashlib
import random
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set
import argparse

from PIL import Image
import numpy as np
from tqdm import tqdm


class CataractDatasetProcessor:
    def __init__(self, raw_data_path: str, output_base_path: str):
        self.raw_data_path = Path(raw_data_path)
        self.output_base_path = Path(output_base_path)
        self.datasets = self._get_dataset_list()
        
    def _get_dataset_list(self) -> List[str]:
        """Get list of available datasets"""
        datasets = []
        for item in self.raw_data_path.iterdir():
            if item.is_dir() and item.name.startswith(('0', '1')):
                datasets.append(item.name)
        return sorted(datasets)
    
    def step1_dedup(self, force: bool = False):
        """Step 1: Remove duplicate images"""
        print("=== Step 1: Deduplication ===")
        
        dedup_path = self.output_base_path / "step1_dedup"
        if dedup_path.exists() and not force:
            print(f"Step 1 output already exists at {dedup_path}")
            print("Use --force to overwrite")
            return dedup_path
            
        # Copy raw data to dedup directory
        for dataset in tqdm(self.datasets, desc="Copying datasets"):
            src_path = self.raw_data_path / dataset
            dst_path = dedup_path / dataset
            
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        
        # Remove duplicates from each dataset
        for dataset in tqdm(self.datasets, desc="Removing duplicates"):
            dataset_path = dedup_path / dataset
            self._remove_duplicates_from_dataset(dataset_path)
            
        print(f"Step 1 completed. Output saved to {dedup_path}")
        return dedup_path
    
    def _remove_duplicates_from_dataset(self, dataset_path: Path):
        """Remove duplicate images from a specific dataset"""
        hash_dict = defaultdict(list)
        
        # Collect all images and their hashes
        for img_file in dataset_path.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                try:
                    img = Image.open(img_file).resize((100, 100)).convert("RGB")
                    img_hash = hashlib.md5(img.tobytes()).hexdigest()
                    hash_dict[img_hash].append(img_file)
                except Exception as e:
                    print(f"Error processing {img_file}: {e}")
        
        # Remove duplicates (keep first occurrence)
        removed_count = 0
        for img_hash, file_list in hash_dict.items():
            if len(file_list) > 1:
                # Keep the first file, remove the rest
                for duplicate_file in file_list[1:]:
                    try:
                        duplicate_file.unlink()
                        removed_count += 1
                    except Exception as e:
                        print(f"Error removing {duplicate_file}: {e}")
        
        if removed_count > 0:
            print(f"  {dataset_path.name}: Removed {removed_count} duplicate images")
    
    def step2_crop(self, force: bool = False):
        """Step 2: Crop images to focus on eye regions"""
        print("=== Step 2: Image Cropping ===")
        
        crop_path = self.output_base_path / "step2_crop"
        if crop_path.exists() and not force:
            print(f"Step 2 output already exists at {crop_path}")
            print("Use --force to overwrite")
            return crop_path
            
        dedup_path = self.output_base_path / "step1_dedup"
        if not dedup_path.exists():
            raise FileNotFoundError(f"Step 1 output not found at {dedup_path}")
        
        # Copy dedup data to crop directory
        for dataset in tqdm(self.datasets, desc="Copying datasets"):
            src_path = dedup_path / dataset
            dst_path = crop_path / dataset
            
            if dst_path.exists():
                shutil.rmtree(dst_path)
            shutil.copytree(src_path, dst_path)
        
        # Crop images in each dataset
        for dataset in tqdm(self.datasets, desc="Cropping images"):
            dataset_path = crop_path / dataset
            self._crop_dataset_images(dataset_path)
            
        print(f"Step 2 completed. Output saved to {crop_path}")
        return crop_path
    
    def _crop_dataset_images(self, dataset_path: Path):
        """Crop images in a dataset to focus on eye regions"""
        # This is a placeholder for actual cropping logic
        # You can implement specific cropping algorithms here
        # For now, we'll just copy the images as-is
        
        for img_file in dataset_path.rglob("*"):
            if img_file.is_file() and img_file.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                # TODO: Implement actual cropping logic
                # For now, just keep the image as-is
                pass
    
    def step3_train_test_split(self, train_ratio: float = 0.8, force: bool = False):
        """Step 3: Split data into train and test sets"""
        print("=== Step 3: Train/Test Split ===")
        
        split_path = self.output_base_path / "step3_train_test_split"
        if split_path.exists() and not force:
            print(f"Step 3 output already exists at {split_path}")
            print("Use --force to overwrite")
            return split_path
            
        crop_path = self.output_base_path / "step2_crop"
        if not crop_path.exists():
            raise FileNotFoundError(f"Step 2 output not found at {crop_path}")
        
        # Create train/test structure
        for dataset in tqdm(self.datasets, desc="Creating train/test split"):
            self._create_train_test_split(dataset, crop_path, split_path, train_ratio)
            
        print(f"Step 3 completed. Output saved to {split_path}")
        return split_path
    
    def _create_train_test_split(self, dataset_name: str, crop_path: Path, split_path: Path, train_ratio: float):
        """Create train/test split for a specific dataset"""
        dataset_path = crop_path / dataset_name
        train_path = split_path / dataset_name / "train"
        test_path = split_path / dataset_name / "test"
        
        # Create directories
        train_path.mkdir(parents=True, exist_ok=True)
        test_path.mkdir(parents=True, exist_ok=True)
        
        for category in ['0', '1']:
            category_path = dataset_path / category
            if not category_path.exists():
                continue
                
            train_category_path = train_path / category
            test_category_path = test_path / category
            train_category_path.mkdir(parents=True, exist_ok=True)
            test_category_path.mkdir(parents=True, exist_ok=True)
            
            # Get all images in this category
            image_files = list(category_path.glob("*"))
            
            if not image_files:
                continue
            
            # Check if this dataset has left/right eye pairs
            if self._has_left_right_pairs(image_files):
                # Split by person ID to avoid splitting left/right eyes
                self._split_by_person_id(image_files, train_category_path, test_category_path, train_ratio)
            else:
                # Simple random split
                self._simple_random_split(image_files, train_category_path, test_category_path, train_ratio)
    
    def _has_left_right_pairs(self, image_files: List[Path]) -> bool:
        """Check if dataset has left/right eye pairs"""
        # Check for common left/right indicators in filenames
        left_indicators = ['left', 'L', '_l']
        right_indicators = ['right', 'R', '_r']
        
        has_left = any(any(indicator in f.name.lower() for indicator in left_indicators) for f in image_files)
        has_right = any(any(indicator in f.name.lower() for indicator in right_indicators) for f in image_files)
        
        return has_left and has_right
    
    def _split_by_person_id(self, image_files: List[Path], train_path: Path, test_path: Path, train_ratio: float):
        """Split images by person ID to avoid splitting left/right eyes"""
        # Group images by person ID
        person_groups = defaultdict(list)
        
        for img_file in image_files:
            person_id = self._extract_person_id(img_file.name)
            person_groups[person_id].append(img_file)
        
        # Split person groups
        person_ids = list(person_groups.keys())
        random.shuffle(person_ids)
        
        split_idx = int(len(person_ids) * train_ratio)
        train_person_ids = person_ids[:split_idx]
        test_person_ids = person_ids[split_idx:]
        
        # Copy images to train/test directories
        for person_id in train_person_ids:
            for img_file in person_groups[person_id]:
                shutil.copy2(img_file, train_path / img_file.name)
        
        for person_id in test_person_ids:
            for img_file in person_groups[person_id]:
                shutil.copy2(img_file, test_path / img_file.name)
    
    def _extract_person_id(self, filename: str) -> str:
        """Extract person ID from filename"""
        # This function needs to be customized based on your filename patterns
        # Examples:
        # C101_S1_I1_L.jpg -> C101
        # 0000_0P_0H_0V_left.png -> 0000
        # 00_20250209_left.png -> 00
        
        # Remove file extension
        name = Path(filename).stem
        
        # Try to extract person ID based on common patterns
        if name.startswith('C') and '_' in name:
            # Pattern: C101_S1_I1_L
            return name.split('_')[0]
        elif name.startswith('0') and '_' in name:
            # Pattern: 0000_0P_0H_0V
            return name.split('_')[0]
        elif name.startswith('0') and len(name.split('_')[0]) <= 2:
            # Pattern: 00_20250209
            return name.split('_')[0]
        else:
            # Fallback: use first part before any separator
            return name.split('_')[0] if '_' in name else name
    
    def _simple_random_split(self, image_files: List[Path], train_path: Path, test_path: Path, train_ratio: float):
        """Simple random split for datasets without left/right pairs"""
        random.shuffle(image_files)
        split_idx = int(len(image_files) * train_ratio)
        
        train_files = image_files[:split_idx]
        test_files = image_files[split_idx:]
        
        # Copy files
        for img_file in train_files:
            shutil.copy2(img_file, train_path / img_file.name)
        
        for img_file in test_files:
            shutil.copy2(img_file, test_path / img_file.name)
    
    def run_pipeline(self, force: bool = False, train_ratio: float = 0.8):
        """Run the complete preprocessing pipeline"""
        print("Starting Cataract Dataset Preprocessing Pipeline")
        print(f"Raw data path: {self.raw_data_path}")
        print(f"Output path: {self.output_base_path}")
        print(f"Datasets found: {len(self.datasets)}")
        print()
        
        try:
            # Step 1: Deduplication
            step1_output = self.step1_dedup(force=force)
            
            # Step 2: Cropping
            step2_output = self.step2_crop(force=force)
            
            # Step 3: Train/Test Split
            step3_output = self.step3_train_test_split(train_ratio=train_ratio, force=force)
            
            print("\n=== Pipeline Completed Successfully ===")
            print(f"Final output: {step3_output}")
            
        except Exception as e:
            print(f"Pipeline failed: {e}")
            raise


def main():
    parser = argparse.ArgumentParser(description="Cataract Dataset Preprocessing Pipeline")
    parser.add_argument("--raw_data_path", type=str, default="raw", 
                       help="Path to raw data directory")
    parser.add_argument("--output_path", type=str, default="preprocessed",
                       help="Path to output directory")
    parser.add_argument("--force", action="store_true",
                       help="Force overwrite existing outputs")
    parser.add_argument("--train_ratio", type=float, default=0.8,
                       help="Ratio of training data (default: 0.8)")
    parser.add_argument("--step", type=int, choices=[1, 2, 3],
                       help="Run specific step only (1: dedup, 2: crop, 3: split)")
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(42)
    
    processor = CataractDatasetProcessor(args.raw_data_path, args.output_path)
    
    if args.step:
        if args.step == 1:
            processor.step1_dedup(force=args.force)
        elif args.step == 2:
            processor.step2_crop(force=args.force)
        elif args.step == 3:
            processor.step3_train_test_split(force=args.force, train_ratio=args.train_ratio)
    else:
        processor.run_pipeline(force=args.force, train_ratio=args.train_ratio)


if __name__ == "__main__":
    main()

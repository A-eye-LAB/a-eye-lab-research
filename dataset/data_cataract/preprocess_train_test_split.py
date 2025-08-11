#!/usr/bin/env python3
"""
Step 4: Train-Test Split
This script splits the cropped dataset into train and test sets while ensuring:
1. For random split datasets (01-06): Files are randomly distributed
2. For person-aware datasets (07-11): Same person's eyes don't get split between train and test
3. Output structure: step4_train_test_split/train/0,1 and test/0,1 with dataset prefix in filenames
"""

import os
import re
import yaml
import shutil
import random
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def load_config(config_path: str) -> dict:
    """Load YAML configuration file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config file: {e}")
        return {}


def extract_person_id(filename: str, regex_pattern: str) -> str:
    """Extract person ID from filename using regex pattern"""
    try:
        match = re.search(regex_pattern, filename)
        if match:
            return match.group(1)
        return None
    except Exception:
        return None


def process_random_split_dataset(source_path: Path, target_train: Path, target_test: Path, 
                               dataset_name: str, train_ratio: float, subdir: str):
    """Process dataset with random split (01-06)"""
    source_subdir = source_path / subdir
    if not source_subdir.exists():
        return 0, 0
    
    files = list(source_subdir.rglob('*'))
    image_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    if not image_files:
        return 0, 0
    
    # Random shuffle
    random.shuffle(image_files)
    
    # Split files
    split_point = int(len(image_files) * train_ratio)
    train_files = image_files[:split_point]
    test_files = image_files[split_point:]
    
    # Copy train files
    for file_path in train_files:
        new_filename = f"{dataset_name}_{file_path.name}"
        target_file = target_train / subdir / new_filename
        shutil.copy2(file_path, target_file)
    
    # Copy test files
    for file_path in test_files:
        new_filename = f"{dataset_name}_{file_path.name}"
        target_file = target_test / subdir / new_filename
        shutil.copy2(file_path, target_file)
    
    return len(train_files), len(test_files)


def process_person_aware_dataset(source_path: Path, target_train: Path, target_test: Path,
                                dataset_name: str, train_ratio: float, person_id_regex: str, subdir: str):
    """Process dataset with person-aware split (07-11)"""
    source_subdir = source_path / subdir
    if not source_subdir.exists():
        return 0, 0
    
    files = list(source_subdir.rglob('*'))
    image_files = [f for f in files if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']]
    
    if not image_files:
        return 0, 0
    
    # Group files by person ID
    person_files = defaultdict(list)
    for file_path in image_files:
        person_id = extract_person_id(file_path.name, person_id_regex)
        if person_id:
            person_files[person_id].append(file_path)
    
    if not person_files:
        return 0, 0
    
    # Split persons (not files)
    person_ids = list(person_files.keys())
    random.shuffle(person_ids)
    
    split_point = int(len(person_ids) * train_ratio)
    train_persons = person_ids[:split_point]
    test_persons = person_ids[split_point:]
    
    # Copy all files for train persons to train
    train_count = 0
    for person_id in train_persons:
        for file_path in person_files[person_id]:
            new_filename = f"{dataset_name}_{file_path.name}"
            target_file = target_train / subdir / new_filename
            shutil.copy2(file_path, target_file)
            train_count += 1
    
    # Copy all files for test persons to test
    test_count = 0
    for person_id in test_persons:
        for file_path in person_files[person_id]:
            new_filename = f"{dataset_name}_{file_path.name}"
            target_file = target_test / subdir / new_filename
            shutil.copy2(file_path, target_file)
            test_count += 1
    
    return train_count, test_count


def main():
    parser = argparse.ArgumentParser(description="Step 4: Train-Test Split")
    parser.add_argument("--source", default="preprocessed/step3_crop", help="Source directory containing cropped images")
    parser.add_argument("--target", default="preprocessed/step4_train_test_split", help="Target directory for train-test split")
    parser.add_argument("--config", default="train_test_split_config.yaml", help="Path to YAML config file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    print("=== Step 4: Train-Test Split ===")
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Config: {args.config}")
    print(f"Random seed: {args.seed}")
    
    # Set random seed
    random.seed(args.seed)
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        return
    
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    if not source_path.exists():
        print(f"Error: Source path {source_path} does not exist")
        return
    
    # Create target directories
    train_path = target_path / "train"
    test_path = target_path / "test"
    
    for subdir in ['0', '1']:
        (train_path / subdir).mkdir(parents=True, exist_ok=True)
        (test_path / subdir).mkdir(parents=True, exist_ok=True)
    
    datasets_config = config.get('datasets', {})
    
    # Process each dataset
    total_stats = {
        'train_files': 0,
        'test_files': 0,
        'datasets_processed': 0
    }
    
    print(f"\nProcessing {len(datasets_config)} datasets...")
    
    for dataset_name in sorted(datasets_config.keys()):
        dataset_path = source_path / dataset_name
        if not dataset_path.exists():
            print(f"\nWarning: Dataset {dataset_name} not found in {source_path}")
            continue
        
        dataset_config = datasets_config[dataset_name]
        split_method = dataset_config.get('split_method', 'random')
        train_ratio = dataset_config['train_ratio']
        
        print(f"\nProcessing {dataset_name} ({split_method} split, {train_ratio:.0%}:{1-train_ratio:.0%})")
        
        dataset_train_count = 0
        dataset_test_count = 0
        
        # Process both 0 and 1 subdirectories
        for subdir in ['0', '1']:
            if split_method == 'random':
                train_count, test_count = process_random_split_dataset(
                    dataset_path, train_path, test_path, dataset_name, train_ratio, subdir
                )
            else:  # person_aware
                person_id_regex = dataset_config['person_id_regex']
                train_count, test_count = process_person_aware_dataset(
                    dataset_path, train_path, test_path, dataset_name, train_ratio, person_id_regex, subdir
                )
            
            dataset_train_count += train_count
            dataset_test_count += test_count
        
        print(f"  {dataset_name}: {dataset_train_count} train, {dataset_test_count} test files")
        
        total_stats['train_files'] += dataset_train_count
        total_stats['test_files'] += dataset_test_count
        total_stats['datasets_processed'] += 1
    
    # Print final summary
    print(f"\n=== Final Summary ===")
    print(f"Datasets processed: {total_stats['datasets_processed']}")
    print(f"Total train files: {total_stats['train_files']}")
    print(f"Total test files: {total_stats['test_files']}")
    print(f"Results saved in: {target_path}")
    
    # Save processing log
    log_file = target_path / "split_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Train-Test Split Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total train files: {total_stats['train_files']}\n")
        f.write(f"Total test files: {total_stats['test_files']}\n")
        f.write(f"Datasets processed: {total_stats['datasets_processed']}\n")
        f.write(f"\nOutput structure:\n")
        f.write(f"- {target_path}/train/0/ (train images, class 0)\n")
        f.write(f"- {target_path}/train/1/ (train images, class 1)\n")
        f.write(f"- {target_path}/test/0/ (test images, class 0)\n")
        f.write(f"- {target_path}/test/1/ (test images, class 1)\n")
        f.write(f"\nFile naming: {dataset_name}_original_filename.ext\n")


if __name__ == "__main__":
    main()

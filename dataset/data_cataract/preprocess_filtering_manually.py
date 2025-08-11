#!/usr/bin/env python3
"""
Process deduplicated dataset files (step2 filtering):
1. Maintain source directory structure
2. If a matching file exists in manually_preprocessed, copy directly to the target directory
3. If no matching file exists, move to filter_out directory within the same structure
"""

import os
import shutil
import argparse
import hashlib
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm


def calculate_file_hash(file_path):
    """Calculate MD5 hash of file content"""
    try:
        with open(file_path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except Exception as e:
        print(f"Error calculating hash for {file_path}: {e}")
        return None


def is_image_file(file_path):
    """Check if file is an image based on extension"""
    extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
    return file_path.suffix.lower() in {ext.lower() for ext in extensions}


def collect_files_by_hash(directory_path):
    """Collect files grouped by file hash"""
    hash_to_files = defaultdict(list)
    
    for file_path in directory_path.rglob('*'):
        if file_path.is_file() and is_image_file(file_path):
            file_hash = calculate_file_hash(file_path)
            if file_hash:
                hash_to_files[file_hash].append(file_path)
    
    return hash_to_files


def main():
    parser = argparse.ArgumentParser(description="Process raw dataset files and separate unmatched ones")
    parser.add_argument("--source", default="preprocessed/step1_dedup_0.7", help="Source directory path (deduplicated data)")
    parser.add_argument("--target", default="preprocessed/step2_manually_filtered", help="Target directory path for filtered files")
    parser.add_argument("--manual", default="manually_preprocessed", help="Manually preprocessed dataset directory path")
    
    args = parser.parse_args()
    
    print("=== Processing Deduplicated Dataset Files ===")
    print(f"Source dataset: {args.source}")
    print(f"Manual dataset: {args.manual}")
    print(f"Target: {args.target}")
    
    source_path = Path(args.source)
    manual_path = Path(args.manual)
    target_path = Path(args.target)
    
    if not source_path.exists():
        print(f"Error: Source dataset path {source_path} does not exist")
        return
    
    if not manual_path.exists():
        print(f"Error: Manual dataset path {manual_path} does not exist")
        return
    
    # Clean target directory
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True)
    
    # Create top-level filter_out directory
    filter_out_path = target_path / "filter_out"
    filter_out_path.mkdir(parents=True, exist_ok=True)
    
    # Collect manual files by hash
    print("Collecting manual dataset files by hash...")
    manual_hash_map = {}
    for subdir in ['0', '1']:
        subdir_path = manual_path / subdir
        if subdir_path.exists():
            for file_path in subdir_path.rglob('*'):
                if file_path.is_file() and is_image_file(file_path):
                    file_hash = calculate_file_hash(file_path)
                    if file_hash:
                        manual_hash_map[file_hash] = file_path
    
    print(f"Found {len(manual_hash_map)} unique files in manual dataset")
    
    # Process source files
    print("Processing source dataset files...")
    matched_count = 0
    filtered_count = 0
    
    # Process datasets 01-11 (01-06: filtered, 07-11: copied directly)
    for dataset_dir in sorted(source_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
            
        dataset_name = dataset_dir.name
        if not dataset_name.startswith(('01_', '02_', '03_', '04_', '05_', '06_', '07_', '08_', '09_', '10_', '11_')):
            continue
            
        print(f"\nProcessing dataset: {dataset_name}")
        
        # Create target directory for this dataset
        target_dataset_dir = target_path / dataset_name
        target_dataset_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if this dataset should be filtered (01-06) or copied directly (07-11)
        should_filter = dataset_name.startswith(('01_', '02_', '03_', '04_', '05_', '06_'))
        
        for subdir in ['0', '1']:
            subdir_path = dataset_dir / subdir
            if not subdir_path.exists():
                continue
            
            # Create target subdirectory
            target_subdir = target_dataset_dir / subdir
            target_subdir.mkdir(parents=True, exist_ok=True)
            
            for source_file in subdir_path.rglob('*'):
                if not source_file.is_file() or not is_image_file(source_file):
                    continue
                    
                if should_filter:
                    # Apply filtering for datasets 01-06
                    file_hash = calculate_file_hash(source_file)
                    if not file_hash:
                        continue
                    
                    if file_hash in manual_hash_map:
                        # Found a match - copy source file directly to target directory
                        try:
                            source_target = target_subdir / source_file.name
                            shutil.copy2(source_file, source_target)
                            
                            matched_count += 1
                            print(f"  Matched: {source_file.name}")
                            
                        except Exception as e:
                            print(f"    Error copying matched file {source_file.name}: {e}")
                    
                    else:
                        # No match found - copy to filter_out
                        try:
                            # Copy source file to filter_out maintaining dataset structure
                            filter_target = filter_out_path / dataset_name / subdir / source_file.name
                            filter_target.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copy2(source_file, filter_target)
                            
                            filtered_count += 1
                            print(f"  Filtered: {source_file.name}")
                            
                        except Exception as e:
                            print(f"    Error copying filtered file {source_file.name}: {e}")
                
                else:
                    # Copy directly for datasets 07-11 without filtering
                    try:
                        source_target = target_subdir / source_file.name
                        shutil.copy2(source_file, source_target)
                        
                        matched_count += 1
                        print(f"  Copied directly: {source_file.name}")
                        
                    except Exception as e:
                        print(f"    Error copying file {source_file.name}: {e}")
    
    print(f"\n=== Summary ===")
    print(f"Total files matched: {matched_count}")
    print(f"Total files filtered out: {filtered_count}")
    print(f"Results saved in: {target_path}")
    
    # Save processing details to log
    log_file = target_path / "processing_log.txt"
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"Source Dataset Processing Log\n")
        f.write("=" * 60 + "\n")
        f.write(f"Total files matched: {matched_count}\n")
        f.write(f"Total files filtered out: {filtered_count}\n")
        f.write(f"\nProcessed files are organized in: {target_path}\n")
        f.write(f"Matched files are directly in dataset directories\n")
        f.write(f"Filtered files are in filter_out directories\n")


if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Deduplication preprocessing combining histogram similarity and current logic
Creates deduplicated data structure in step1_dedup folder
"""

import hashlib
import os
import shutil
import argparse
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def calculate_image_hash(file_path):
    """Calculate hash of resized image (100x100) - from current logic"""
    try:
        img = Image.open(file_path).resize((100, 100)).convert("RGB")
        img_hash = hashlib.md5(img.tobytes()).hexdigest()
        return img_hash
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def calculate_histogram_features(file_path):
    """Calculate histogram-based features for similarity"""
    try:
        img = Image.open(file_path).convert("RGB")
        
        # RGB histogram (normalized)
        rgb_hist = img.histogram()
        rgb_hist_norm = np.array(rgb_hist) / sum(rgb_hist)
        
        # Grayscale histogram (normalized)
        gray_img = img.convert('L')
        gray_hist = gray_img.histogram()
        gray_hist_norm = np.array(gray_hist) / sum(gray_hist)
        
        # Color moments
        img_array = np.array(img)
        color_moments = []
        for channel in range(3):
            channel_data = img_array[:, :, channel]
            color_moments.extend([np.mean(channel_data), np.std(channel_data)])
        
        return {
            'rgb_hist': rgb_hist_norm,
            'gray_hist': gray_hist_norm,
            'color_moments': color_moments
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def calculate_similarity(features1, features2):
    """Calculate similarity between two feature sets"""
    if not features1 or not features2:
        return 0.0
    
    similarities = []
    
    # RGB histogram similarity
    try:
        rgb_sim = cosine_similarity(
            features1['rgb_hist'].reshape(1, -1), 
            features2['rgb_hist'].reshape(1, -1)
        )[0, 0]
        similarities.append(rgb_sim)
    except:
        similarities.append(0.0)
    
    # Grayscale histogram similarity
    try:
        gray_sim = cosine_similarity(
            features1['gray_hist'].reshape(1, -1), 
            features2['gray_hist'].reshape(1, -1)
        )[0, 0]
        similarities.append(gray_sim)
    except:
        similarities.append(0.0)
    
    # Color moments similarity
    try:
        cm1 = np.array(features1['color_moments'])
        cm2 = np.array(features2['color_moments'])
        cm_dist = np.linalg.norm(cm1 - cm2)
        cm_sim = 1.0 / (1.0 + cm_dist)
        similarities.append(cm_sim)
    except:
        similarities.append(0.0)
    
    return np.mean(similarities) if similarities else 0.0


def find_duplicates_combined(files, similarity_threshold=0.7):
    """Find duplicates using both current logic AND histogram similarity"""
    # Calculate both hashes and features
    image_data = {}
    for file_path in tqdm(files, desc="Processing images"):
        img_hash = calculate_image_hash(file_path)
        hist_features = calculate_histogram_features(file_path)
        
        if img_hash and hist_features:
            image_data[file_path] = {
                'hash': img_hash,
                'features': hist_features
            }
    
    # Find duplicates using combined approach
    duplicates = []
    processed = set()
    representative_groups = []
    
    for file_path in tqdm(image_data.keys(), desc="Finding duplicates"):
        if file_path in processed:
            continue
        
        current_data = image_data[file_path]
        current_group = [file_path]
        processed.add(file_path)
        
        # Check if belongs to existing group
        added_to_existing = False
        for rep_data, group_files in representative_groups:
            # Check both hash AND histogram similarity
            hash_match = (current_data['hash'] == rep_data['hash'])
            hist_sim = calculate_similarity(current_data['features'], rep_data['features'])
            hist_match = (hist_sim >= similarity_threshold)
            
            # Use OR logic: either hash matches OR histogram is similar
            if hash_match or hist_match:
                group_files.append(file_path)
                processed.add(file_path)
                added_to_existing = True
                break
        
        if not added_to_existing:
            representative_groups.append((current_data, current_group))
    
    # Convert to duplicates format
    for rep_data, group_files in representative_groups:
        if len(group_files) > 1:
            duplicates.append(group_files)
    
    return duplicates, representative_groups


def process_dataset(source_path, target_path, similarity_threshold=0.7):
    """Process single dataset and create deduplicated structure preserving 0/1 structure"""
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    print(f"\nProcessing: {source_path.name}")
    
    # Process 0 and 1 subdirectories separately to maintain structure
    total_stats = {"total": 0, "duplicates": 0, "groups": 0, "representatives": 0, "removed_duplicates": 0}
    all_duplicate_groups = []  # Store all duplicate groups for logging
    
    for subdir in ['0', '1']:
        subdir_path = source_path / subdir
        if not subdir_path.exists() or not subdir_path.is_dir():
            continue
            
        # Collect all files and separate images from non-images
        image_files = []
        other_files = []
        for file_path in subdir_path.rglob("*"):
            if file_path.is_file():
                if file_path.suffix.lower() in ('.jpg', '.jpeg', '.png', '.webp'):
                    image_files.append(file_path)
                else:
                    other_files.append(file_path)
        
        # Copy non-image files directly
        for other_file in other_files:
            relative_path = other_file.relative_to(subdir_path)
            target_file = target_path / source_path.name / subdir / relative_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            try:
                shutil.copy2(other_file, target_file)
            except Exception as e:
                print(f"Error copying non-image file {other_file}: {e}")
        
        if not image_files:
            continue
        
        # Find duplicates
        duplicates, representative_groups = find_duplicates_combined(image_files, similarity_threshold)
        
        # Store duplicate groups with subdirectory info for logging
        for group in representative_groups:
            if len(group[1]) > 1:  # Only groups with duplicates
                all_duplicate_groups.append({
                    'subdir': subdir,
                    'group_files': group[1],
                    'representative': group[1][0]
                })
        
        # Create target subdirectory structure
        target_subdir = target_path / source_path.name / subdir
        target_subdir.mkdir(parents=True, exist_ok=True)
        
        # Copy representative and duplicates to group folders
        representatives_copied = 0
        removed_duplicates = 0
        duplicated_removed_dir = target_path / "duplicated_removed" / source_path.name / subdir
        
        for group_idx, (rep_data, group_files) in enumerate(representative_groups, 1):
            if len(group_files) > 1:  # Only process groups with duplicates
                # Create group directory
                group_dir = duplicated_removed_dir / f"group_{group_idx:03d}"
                group_dir.mkdir(parents=True, exist_ok=True)
                
                # Copy representative with prefix
                rep_file = group_files[0]
                rep_target = group_dir / f"representative_{rep_file.name}"
                try:
                    shutil.copy2(rep_file, rep_target)
                    representatives_copied += 1
                except Exception as e:
                    print(f"    Error copying representative {rep_file}: {e}")
                
                # Copy duplicates to the same group folder
                for duplicate_file in group_files[1:]:
                    duplicate_target = group_dir / duplicate_file.name
                    try:
                        shutil.copy2(duplicate_file, duplicate_target)
                        removed_duplicates += 1
                    except Exception as e:
                        print(f"    Error copying duplicate {duplicate_file}: {e}")
            elif group_files:  # Single file without duplicates - copy to main target dir
                rep_file = group_files[0]
                target_file = target_subdir / rep_file.name
                try:
                    shutil.copy2(rep_file, target_file)
                    representatives_copied += 1
                except Exception as e:
                    print(f"    Error copying {rep_file}: {e}")
        
        # Update total statistics
        total_stats["total"] += len(image_files)
        total_stats["duplicates"] += sum(len(group) for group in duplicates)
        total_stats["groups"] += len(duplicates)
        total_stats["representatives"] += representatives_copied
        total_stats["removed_duplicates"] += removed_duplicates
    
    print(f"  Results: {total_stats['total']} images, {total_stats['duplicates']} duplicates, {total_stats['representatives']} representatives")
    
    # Create log directory and save detailed log
    log_dir = target_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{source_path.name}_deduplication_log.txt"
    log_deduplication_details(log_file, source_path.name, all_duplicate_groups, total_stats)
    
    return total_stats


def log_deduplication_details(log_file, dataset_name, duplicate_groups, stats):
    """Log detailed deduplication information to txt file"""
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(f"Deduplication Log for {dataset_name}\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {Path.cwd()}\n")
            f.write(f"Similarity Threshold: 0.8\n\n")
            
            # Summary statistics
            f.write("SUMMARY STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Images Processed: {stats['total']}\n")
            f.write(f"Total Duplicates Found: {stats['duplicates']}\n")
            f.write(f"Duplicate Groups: {stats['groups']}\n")
            f.write(f"Representatives Saved: {stats['representatives']}\n")
            f.write(f"Duplicates Removed: {stats['removed_duplicates']}\n")
            f.write(f"Reduction Rate: {((stats['duplicates'] - stats['representatives']) / stats['duplicates'] * 100):.1f}%\n\n")
            
            # Detailed duplicate groups
            if duplicate_groups:
                f.write("DETAILED DUPLICATE GROUPS\n")
                f.write("-" * 40 + "\n")
                
                for i, group_info in enumerate(duplicate_groups, 1):
                    f.write(f"\nGroup {i} ({group_info['subdir']}/):\n")
                    f.write(f"  Representative: {group_info['representative'].name}\n")
                    f.write(f"  Total in group: {len(group_info['group_files'])}\n")
                    f.write("  All files:\n")
                    
                    for j, file_path in enumerate(group_info['group_files']):
                        if j == 0:
                            f.write(f"    → {file_path.name} (REPRESENTATIVE)\n")
                        else:
                            f.write(f"    - {file_path.name} (DUPLICATE)\n")
                    
                    f.write(f"  Duplicates removed: {len(group_info['group_files']) - 1}\n")
            else:
                f.write("No duplicates found in this dataset.\n")
            
            f.write(f"\n" + "=" * 60 + "\n")
            f.write("End of Log\n")
        
        print(f"  📝 Detailed log saved to: {log_file}")
        
    except Exception as e:
        print(f"  Error writing log file: {e}")


def create_overall_summary_log(log_file, all_stats, threshold):
    """Create overall summary log for all datasets"""
    try:
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write("OVERALL DEDUPLICATION SUMMARY LOG\n")
            f.write("=" * 60 + "\n")
            f.write(f"Generated: {Path.cwd()}\n")
            f.write(f"Similarity Threshold: {threshold}\n\n")
            
            # Overall statistics
            total_images = sum(stats['total'] for stats in all_stats.values())
            total_duplicates = sum(stats['duplicates'] for stats in all_stats.values())
            total_representatives = sum(stats['representatives'] for stats in all_stats.values())
            total_removed = sum(stats['removed_duplicates'] for stats in all_stats.values())
            
            f.write("OVERALL STATISTICS\n")
            f.write("-" * 30 + "\n")
            f.write(f"Total Datasets Processed: {len(all_stats)}\n")
            f.write(f"Total Images Processed: {total_images}\n")
            f.write(f"Total Duplicates Found: {total_duplicates}\n")
            f.write(f"Total Representatives Saved: {total_representatives}\n")
            f.write(f"Total Duplicates Removed: {total_removed}\n")
            f.write(f"Overall Reduction Rate: {((total_duplicates - total_representatives) / total_duplicates * 100):.1f}%\n\n")
            
            # Per-dataset breakdown
            f.write("PER-DATASET BREAKDOWN\n")
            f.write("-" * 40 + "\n")
            f.write(f"{'Dataset':<30} {'Total':<8} {'Duplicates':<12} {'Representatives':<15} {'Removed':<10}\n")
            f.write("-" * 80 + "\n")
            
            for dataset, stats in all_stats.items():
                f.write(f"{dataset:<30} {stats['total']:<8} {stats['duplicates']:<12} {stats['representatives']:<15} {stats['removed_duplicates']:<10}\n")
            
            f.write(f"\n" + "=" * 60 + "\n")
            f.write("End of Overall Summary Log\n")
        
    except Exception as e:
        print(f"  Error writing overall summary log: {e}")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate images using combined logic")
    parser.add_argument("--source", default="raw", help="Source directory path")
    parser.add_argument("--target", default="step1_dedup", help="Target directory path")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold")

    
    args = parser.parse_args()
    
    print("=== Starting Deduplication ===")
    print(f"Source: {args.source}, Target: {args.target}, Threshold: {args.threshold}")
    
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    # Clean target directory
    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True)
    
    # Process all datasets
    all_stats = []
    total_images = 0
    total_duplicates = 0
    total_representatives = 0
    total_removed = 0
    
    # Define dataset groups
    dedup_datasets = [
        "01_pub_alexandra",
        "02_pub_hemooredaoo",
        "03_pub_mohammedgamal37130",
        "04_pub_krishnabojha",
        "05_pub_piygot5",
        "06_pub_kershrita"
    ]
    
    copy_datasets = [
        "07_pub_UBIPr",
        "08_pub_DIRL",
        "09_collected_india_data_app",
        "10_collected_india_field_trip",
        "11_collected_kor_data"
    ]
    
    for dataset_dir in sorted(source_path.iterdir()):
        dataset_name = dataset_dir.name
        if not dataset_dir.is_dir():
            continue
            
        # Skip if not in either list
        if dataset_name not in dedup_datasets and dataset_name not in copy_datasets:
            continue
            
        # For datasets 07 and above, just copy the directory structure
        if dataset_name in copy_datasets:
            target_dataset_dir = target_path / dataset_name
            if target_dataset_dir.exists():
                shutil.rmtree(target_dataset_dir)
            shutil.copytree(dataset_dir, target_dataset_dir)
            print(f"\nCopied dataset without deduplication: {dataset_name}")
            continue
        
        print(f"\nStarting to process dataset: {dataset_dir.name}")
        stats = process_dataset(dataset_dir, target_path, args.threshold)
        all_stats.append((dataset_dir.name, stats))
        
        total_images += stats['total']
        total_duplicates += stats['duplicates']
        total_representatives += stats['representatives']
        total_removed += stats['removed_duplicates']
    
    # Summary
    print("\n=== Final Summary ===")
    print(f"Total images: {total_images}")
    print(f"Duplicates found: {total_duplicates}")
    print(f"Representatives saved: {total_representatives}")
    print(f"Duplicates removed: {total_removed}")
    print(f"Results saved in: {target_path}")
    
    # Create overall summary log in log directory
    log_dir = target_path / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary_log_file = log_dir / "OVERALL_SUMMARY_LOG.txt"
    create_overall_summary_log(summary_log_file, all_stats, args.threshold)
    print(f"Overall summary log saved to: {summary_log_file}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Step 3: Iris Detection and Cropping
This script processes the manually filtered images from step2_manually_filtered,
detects iris regions using YOLO model, and crops the images accordingly.
Failed detections (no detection or multiple detections) are moved to filter_out.
"""

from ultralytics import YOLO
import cv2
import numpy as np
import os
import shutil
import argparse
from pathlib import Path
from tqdm import tqdm


def crop_iris(
        model,
        image_path,
        conf_threshold=0.25,
        is_box=False,
):
    """Process single image to detect and crop iris region"""
    # Read image
    image = cv2.imread(str(image_path))
    if image is None:
        return None, "no_detection"

    # Get image dimensions
    img_height, img_width = image.shape[:2]

    # Make prediction
    results = model(image, conf=conf_threshold, verbose=False)[0]
    boxes = results.boxes.data.tolist()

    if not boxes:
        return None, "no_detection"

    # Check if multiple detections
    if len(boxes) > 1:
        return None, "multiple_detection"

    # Single detection - find the box with highest confidence score
    best_box = max(boxes, key=lambda x: x[4])
    x1, y1, x2, y2, score, class_id = best_box

    # Convert to integers for drawing
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    if is_box:
        cv2.rectangle(image, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Calculate center and crop size
    center_x = (x1 + x2) / 2
    center_y = (y1 + y2) / 2

    # Use larger side for square crop
    rect_size = max(x2-x1, y2-y1)
    crop_size = min(img_width, img_height, rect_size * 3)

    # Calculate crop coordinates
    x1 = center_x - (crop_size / 2)
    x2 = center_x + (crop_size / 2)
    y1 = center_y - (crop_size / 2)
    y2 = center_y + (crop_size / 2)

    # Adjust coordinates if they go outside image bounds
    if x1 < 0:
        x2 += (0 - x1)
        x1 = 0
    if y1 < 0:
        y2 += (0 - y1)
        y1 = 0
    if x2 > img_width:
        x1 -= (x2 - img_width)
        x2 = img_width
    if y2 > img_height:
        y1 -= (y2 - img_height)
        y2 = img_height

    # Convert to integers for cropping
    x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

    # Crop image
    crop_image = image[y1:y2, x1:x2]
    
    return crop_image, "success"


def process_dataset(
        model,
        source_path,
        target_path,
        conf_threshold=0.25,
        is_box=False,
        save_label=False
):
    """Process a single dataset directory"""
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    print(f"\nProcessing dataset: {source_path.name}")
    
    # Process main dataset directory (non-duplicates)
    stats = {"total": 0, "processed": 0, "failed": 0, "no_detection": 0, "multiple_detection": 0}
    
    # Process both 0 and 1 subdirectories
    for subdir in ['0', '1']:
        source_subdir = source_path / subdir
        if not source_subdir.exists() or not source_subdir.is_dir():
            continue
            
        # Create target subdirectory
        target_subdir = target_path / source_path.name / subdir
        target_subdir.mkdir(parents=True, exist_ok=True)
        
        # Create filter_out subdirectory for failed detections
        filter_out_subdir = target_path / "filter_out" / source_path.name / subdir
        filter_out_subdir.mkdir(parents=True, exist_ok=True)
        
        # Process all images in the directory
        for img_file in tqdm(list(source_subdir.rglob("*")), desc=f"Processing {subdir}"):
            if not img_file.is_file() or not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                continue
                
            stats["total"] += 1
            
            try:
                # Crop iris region
                result = crop_iris(model, img_file, conf_threshold, is_box)
                
                if result is None:
                    continue
                    
                cropped_img, status = result
                
                if status == "success":
                    # Save cropped image
                    target_file = target_subdir / img_file.name
                    cv2.imwrite(str(target_file), cropped_img)
                    
                    if save_label:
                        # Save label file
                        label_file = target_subdir / f"{img_file.stem}.txt"
                        with open(label_file, 'w') as f:
                            f.write("0 0.5 0.5 1.0 1.0\n")  # Normalized coordinates
                    
                    stats["processed"] += 1
                else:
                    # Move failed detection to filter_out
                    filter_target = filter_out_subdir / img_file.name
                    shutil.copy2(img_file, filter_target)
                    
                    if status == "no_detection":
                        stats["no_detection"] += 1
                    elif status == "multiple_detection":
                        stats["multiple_detection"] += 1
                    
                    stats["failed"] += 1
                    
            except Exception as e:
                # Move error files to filter_out
                filter_target = filter_out_subdir / img_file.name
                shutil.copy2(img_file, filter_target)
                stats["failed"] += 1
    
    print(f"Results for {source_path.name}:")
    print(f"  Total images: {stats['total']}")
    print(f"  Successfully processed: {stats['processed']}")
    print(f"  Failed: {stats['failed']} (No detection: {stats['no_detection']}, Multiple: {stats['multiple_detection']})")
    
    return stats


def copy_dataset_directly(source_path, target_path):
    """Copy a dataset directory directly without processing."""
    source_path = Path(source_path)
    target_path = Path(target_path)
    
    print(f"\nCopying dataset {source_path.name} directly...")
    
    # Create target subdirectory
    target_subdir = target_path / source_path.name
    target_subdir.mkdir(parents=True, exist_ok=True)
    
    # Process both 0 and 1 subdirectories
    stats = {"total": 0, "processed": 0, "failed": 0}
    
    for subdir in ['0', '1']:
        source_subdir = source_path / subdir
        if not source_subdir.exists() or not source_subdir.is_dir():
            continue
            
        # Create target subdirectory
        target_subdir_copy = target_subdir / subdir
        target_subdir_copy.mkdir(parents=True, exist_ok=True)
        
        # Process all images in the directory
        for img_file in tqdm(list(source_subdir.rglob("*")), desc=f"Copying {subdir}"):
            if not img_file.is_file() or not img_file.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                continue
                
            stats["total"] += 1
            
            try:
                # Copy image
                target_file = target_subdir_copy / img_file.name
                shutil.copy2(img_file, target_file)
                stats["processed"] += 1
            except Exception as e:
                # Move error files to filter_out
                filter_target = target_subdir_copy / img_file.name
                shutil.copy2(img_file, filter_target)
                stats["failed"] += 1
    
    print(f"Results for {source_path.name}:")
    print(f"  Total images: {stats['total']}")
    print(f"  Successfully copied: {stats['processed']}")
    print(f"  Failed: {stats['failed']}")
    
    return stats


def main():
    parser = argparse.ArgumentParser(description="Step 3: Iris Detection and Cropping")
    parser.add_argument("--source", default="preprocessed/step2_manually_filtered", help="Source directory containing manually filtered images")
    parser.add_argument("--target", default="preprocessed/step3_crop", help="Target directory for cropped images")
    parser.add_argument("--model", default="models/iris.pt", help="Path to YOLO model")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--box", action="store_true", help="Draw detection box on output images")
    parser.add_argument("--save-label", action="store_true", help="Save label files")
    
    args = parser.parse_args()
    
    print("=== Step 3: Iris Detection and Cropping ===")
    print(f"Source: {args.source}")
    print(f"Target: {args.target}")
    print(f"Model: {args.model}")
    print(f"Confidence threshold: {args.conf}")
    
    # Load YOLO model
    try:
        model = YOLO(args.model)
        # Disable verbose output
        model.verbose = False
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    source_path = Path(args.source)
    target_path = Path(args.target)
    
    # Create target directory
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Process datasets (01-08: iris cropping, 09-11: copied directly)
    total_stats = {"total": 0, "processed": 0, "failed": 0, "no_detection": 0, "multiple_detection": 0}
    
    for dataset_dir in sorted(source_path.iterdir()):
        if not dataset_dir.is_dir():
            continue
            
        # Skip filter_out and log directories from previous steps
        if dataset_dir.name in ['filter_out', 'log']:
            continue
            
        # Process all datasets 01-11
        try:
            dataset_num = int(dataset_dir.name.split('_')[0])
            if dataset_num < 1 or dataset_num > 11:
                continue
        except (ValueError, IndexError):
            continue
        
        if dataset_num >= 9:
            # Copy datasets 09-11 directly without iris processing
            copy_stats = copy_dataset_directly(dataset_dir, target_path)
            total_stats["total"] += copy_stats["total"]
            total_stats["processed"] += copy_stats["processed"]
            total_stats["failed"] += copy_stats["failed"]
        else:
            # Process datasets 01-08 with iris detection and cropping
            stats = process_dataset(
                model,
                dataset_dir,
                target_path,
                args.conf,
                args.box,
                args.save_label
            )
            total_stats["total"] += stats["total"]
            total_stats["processed"] += stats["processed"]
            total_stats["failed"] += stats["failed"]
            total_stats["no_detection"] += stats["no_detection"]
            total_stats["multiple_detection"] += stats["multiple_detection"]
    
    # Print final summary
    print("\n=== Final Summary ===")
    print(f"Total images processed: {total_stats['total']}")
    print(f"Successfully cropped: {total_stats['processed']}")
    print(f"Failed: {total_stats['failed']}")
    print(f"  - No detection: {total_stats['no_detection']}")
    print(f"  - Multiple detections: {total_stats['multiple_detection']}")
    print(f"Success rate: {(total_stats['processed']/total_stats['total']*100):.1f}%")
    print(f"\nResults saved in: {target_path}")
    print(f"Failed detections moved to: {target_path}/filter_out")


if __name__ == "__main__":
    main()

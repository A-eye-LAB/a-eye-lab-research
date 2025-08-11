#!/usr/bin/env python3
"""
Dataset Statistics Analyzer
This script analyzes the statistics of each dataset including:
- Total file count per dataset
- Unique person count per dataset  
- Files per person distribution
- Expected train/test split results
"""

import os
import re
import yaml
import argparse
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def load_config(config_path: str) -> Dict:
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


def analyze_dataset(dataset_path: Path, dataset_config: Dict) -> Dict:
    """Analyze a single dataset"""
    stats = {
        'total_files': 0,
        'unique_persons': 0,
        'files_per_person': {},
        'person_distribution': Counter(),
        'expected_train_test': {},
        'split_method': dataset_config.get('split_method', 'random')
    }
    
    person_files = defaultdict(list)
    
    # Debug: print split method and regex pattern
    print(f"    Debug: Split method: {stats['split_method']}")
    if stats['split_method'] == 'person_aware':
        print(f"    Debug: Using regex pattern: {dataset_config['person_id_regex']}")
    
    # Process both 0 and 1 subdirectories
    for subdir in ['0', '1']:
        subdir_path = dataset_path / subdir
        if not subdir_path.exists():
            continue
            
        for file_path in subdir_path.rglob('*'):
            if not file_path.is_file() or not file_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.webp']:
                continue
                
            stats['total_files'] += 1
            filename = file_path.name
            
            if stats['split_method'] == 'person_aware':
                # Extract person ID for person-aware split
                person_id = extract_person_id(filename, dataset_config['person_id_regex'])
                if person_id:
                    person_files[person_id].append(filename)
                    stats['person_distribution'][person_id] += 1
                else:
                    # Debug: show first few failed extractions
                    if stats['total_files'] <= 3:
                        print(f"    Debug: Failed to extract person ID from: {filename}")
                        print(f"    Debug: Regex pattern: {dataset_config['person_id_regex']}")
                        # Test the regex directly
                        import re
                        match = re.match(dataset_config['person_id_regex'], filename)
                        print(f"    Debug: Direct regex test result: {match}")
                        if match:
                            print(f"    Debug: Match groups: {match.groups()}")
            else:
                # For random split, use filename as person ID (each file is unique)
                person_id = filename
                person_files[person_id].append(filename)
                stats['person_distribution'][person_id] += 1
    
    # Calculate statistics
    stats['unique_persons'] = len(person_files)
    stats['files_per_person'] = dict(stats['person_distribution'])
    
    if stats['unique_persons'] > 0:
        file_counts = list(stats['person_distribution'].values())
        stats['min_files_per_person'] = min(file_counts)
        stats['max_files_per_person'] = max(file_counts)
        stats['avg_files_per_person'] = sum(file_counts) / len(file_counts)
        
        # Calculate expected train/test split
        train_ratio = dataset_config['train_ratio']
        test_ratio = dataset_config['test_ratio']
        
        if stats['split_method'] == 'person_aware':
            # Person-aware split: split by person count
            person_ids = list(person_files.keys())
            train_person_count = int(len(person_ids) * train_ratio)
            test_person_count = len(person_ids) - train_person_count
            
            # Calculate expected file counts
            train_files = sum(stats['person_distribution'][pid] for pid in person_ids[:train_person_count])
            test_files = sum(stats['person_distribution'][pid] for pid in person_ids[train_person_count:])
        else:
            # Random split: split by file count directly
            total_files = stats['total_files']
            train_files = int(total_files * train_ratio)
            test_files = total_files - train_files
            train_person_count = train_files
            test_person_count = test_files
        
        stats['expected_train_test'] = {
            'train_persons': train_person_count,
            'test_persons': test_person_count,
            'train_files': train_files,
            'test_files': test_files,
            'train_ratio': train_ratio,
            'test_ratio': test_ratio
        }
    
    return stats


def print_dataset_stats(dataset_name: str, stats: Dict, dataset_config: Dict):
    """Print formatted statistics for a dataset"""
    print(f"\n{dataset_name}:")
    print(f"  - Split method: {stats['split_method']}")
    print(f"  - Total files: {stats['total_files']}")
    
    if stats['split_method'] == 'person_aware':
        print(f"  - Unique persons: {stats['unique_persons']}")
        
        if stats['unique_persons'] > 0:
            print(f"  - Files per person: min={stats['min_files_per_person']}, "
                  f"max={stats['max_files_per_person']}, avg={stats['avg_files_per_person']:.1f}")
            
            expected = stats['expected_train_test']
            print(f"  - Train/Test split ({expected['train_ratio']:.0%}:{expected['test_ratio']:.0%}): "
                  f"{expected['train_persons']} persons → {expected['train_files']} files / "
                  f"{expected['test_persons']} persons → {expected['test_files']} files")
            
            # Show person distribution (top 5)
            print(f"  - Top 5 persons by file count:")
            sorted_persons = sorted(stats['person_distribution'].items(), key=lambda x: x[1], reverse=True)
            for person_id, count in sorted_persons[:5]:
                print(f"    {person_id}: {count} files")
        else:
            print(f"  - No valid files found or person ID extraction failed")
    else:
        # Random split
        expected = stats['expected_train_test']
        print(f"  - Train/Test split ({expected['train_ratio']:.0%}:{expected['test_ratio']:.0%}): "
              f"{expected['train_files']} files / {expected['test_files']} files")


def main():
    parser = argparse.ArgumentParser(description="Analyze dataset statistics for train-test split planning")
    parser.add_argument("--source", default="preprocessed/step3_crop", help="Source directory containing datasets")
    parser.add_argument("--config", default="train_test_split_config.yaml", help="Path to YAML config file")
    
    args = parser.parse_args()
    
    print("=== Dataset Statistics Analyzer ===")
    print(f"Source: {args.source}")
    print(f"Config: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    if not config:
        return
    
    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Error: Source path {source_path} does not exist")
        return
    
    datasets_config = config.get('datasets', {})
    
    # Analyze each dataset
    total_stats = {
        'total_files': 0,
        'total_persons': 0,
        'datasets_analyzed': 0
    }
    
    print(f"\nAnalyzing {len(datasets_config)} datasets...")
    
    for dataset_name in sorted(datasets_config.keys()):
        dataset_path = source_path / dataset_name
        if not dataset_path.exists():
            print(f"\nWarning: Dataset {dataset_name} not found in {source_path}")
            continue
            
        dataset_config = datasets_config[dataset_name]
        print(f"\nDebug: Processing dataset {dataset_name} with config: {dataset_config}")
        stats = analyze_dataset(dataset_path, dataset_config)
        
        print_dataset_stats(dataset_name, stats, dataset_config)
        
        # Update total statistics
        total_stats['total_files'] += stats['total_files']
        total_stats['total_persons'] += stats['unique_persons']
        total_stats['datasets_analyzed'] += 1
    
    # Print summary
    print(f"\n=== Summary ===")
    print(f"Datasets analyzed: {total_stats['datasets_analyzed']}")
    print(f"Total files: {total_stats['total_files']}")
    print(f"Total unique persons: {total_stats['total_persons']}")
    print(f"Average files per person: {total_stats['total_files'] / total_stats['total_persons']:.1f}" if total_stats['total_persons'] > 0 else "No persons found")


if __name__ == "__main__":
    main()

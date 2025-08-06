#!/usr/bin/env python3
"""
SecureEye Dataset Setup Script
This script creates the necessary directory structure for the SecureEye training dataset.
"""

import os
import yaml
from pathlib import Path
import shutil

def create_dataset_structure():
    """Create the SecureEye dataset directory structure"""
    
    # Define the base path for the dataset
    base_path = Path('../datasets/secureeye')
    
    # Define the directory structure
    directories = [
        base_path / 'train' / 'images',
        base_path / 'train' / 'labels',
        base_path / 'val' / 'images',
        base_path / 'val' / 'labels',
        base_path / 'test' / 'images',
        base_path / 'test' / 'labels'
    ]
    
    print("Creating SecureEye dataset directory structure...")
    
    # Create directories
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✓ Created: {directory}")
    
    # Create README files in each directory
    readme_content = """# SecureEye Dataset Directory

This directory contains the SecureEye surveillance dataset.

## Directory Structure:
- images/: Contains the image files (JPG, PNG)
- labels/: Contains the YOLO format label files (.txt)

## Label Format:
Each label file should contain one line per object in YOLO format:
class_id x_center y_center width height

## Classes:
0: person
1: fire
2: smoke
3: weapon
4: vehicle
5: crowd
6: intrusion
7: violence
8: running
9: falling
10: suspicious_activity
11: unauthorized_access
12: vandalism
13: theft
14: loitering

## Guidelines:
- Use high-quality images (minimum 640x640 pixels)
- Include various lighting conditions
- Include different camera angles
- Ensure accurate bounding box annotations
"""
    
    for split in ['train', 'val', 'test']:
        readme_path = base_path / split / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(readme_content)
        print(f"✓ Created README: {readme_path}")
    
    print(f"\nDataset structure created at: {base_path}")
    return base_path

def create_sample_labels():
    """Create sample label files to demonstrate the format"""
    
    base_path = Path('../datasets/secureeye')
    
    # Sample label content (YOLO format)
    sample_labels = {
        'train': [
            # Sample 1: Person detection
            "0 0.5 0.5 0.2 0.4\n",
            # Sample 2: Fire detection
            "1 0.3 0.7 0.4 0.3\n",
            # Sample 3: Multiple objects
            "0 0.2 0.3 0.15 0.3\n4 0.8 0.6 0.3 0.2\n"
        ],
        'val': [
            # Sample 1: Smoke detection
            "2 0.4 0.4 0.3 0.2\n",
            # Sample 2: Weapon detection
            "3 0.6 0.5 0.1 0.2\n"
        ],
        'test': [
            # Sample 1: Crowd detection
            "5 0.5 0.5 0.6 0.4\n",
            # Sample 2: Violence detection
            "7 0.3 0.4 0.4 0.3\n"
        ]
    }
    
    print("\nCreating sample label files...")
    
    for split, labels in sample_labels.items():
        for i, label_content in enumerate(labels):
            label_path = base_path / split / 'labels' / f'sample_{i+1}.txt'
            with open(label_path, 'w') as f:
                f.write(label_content)
            print(f"✓ Created sample label: {label_path}")
    
    print("\nSample label files created. These demonstrate the YOLO format.")
    print("Replace these with your actual labeled data.")

def create_dataset_info():
    """Create a dataset information file"""
    
    base_path = Path('../datasets/secureeye')
    info_path = base_path / 'dataset_info.yaml'
    
    dataset_info = {
        'name': 'SecureEye Surveillance Dataset',
        'description': 'Dataset for training YOLOv5 models to detect security events in surveillance footage',
        'version': '1.0',
        'created_by': 'SecureEye Team',
        'classes': {
            0: 'person',
            1: 'fire',
            2: 'smoke',
            3: 'weapon',
            4: 'vehicle',
            5: 'crowd',
            6: 'intrusion',
            7: 'violence',
            8: 'running',
            9: 'falling',
            10: 'suspicious_activity',
            11: 'unauthorized_access',
            12: 'vandalism',
            13: 'theft',
            14: 'loitering'
        },
        'expected_distribution': {
            'person': '30%',
            'fire': '5%',
            'smoke': '5%',
            'weapon': '3%',
            'vehicle': '15%',
            'crowd': '10%',
            'intrusion': '8%',
            'violence': '5%',
            'running': '8%',
            'falling': '3%',
            'suspicious_activity': '5%',
            'unauthorized_access': '3%',
            'vandalism': '2%',
            'theft': '2%',
            'loitering': '5%'
        },
        'image_requirements': {
            'min_resolution': '640x640',
            'formats': ['jpg', 'jpeg', 'png'],
            'aspect_ratio': 'flexible'
        },
        'label_format': 'YOLO (class_id x_center y_center width height)',
        'splits': {
            'train': '70%',
            'val': '20%',
            'test': '10%'
        }
    }
    
    with open(info_path, 'w') as f:
        yaml.dump(dataset_info, f, default_flow_style=False, sort_keys=False)
    
    print(f"✓ Created dataset info: {info_path}")

def main():
    """Main function to set up the dataset"""
    
    print("=" * 60)
    print("SecureEye Dataset Setup")
    print("=" * 60)
    
    # Create directory structure
    base_path = create_dataset_structure()
    
    # Create sample labels
    create_sample_labels()
    
    # Create dataset info
    create_dataset_info()
    
    print("\n" + "=" * 60)
    print("Setup Complete!")
    print("=" * 60)
    print(f"\nDataset location: {base_path}")
    print("\nNext steps:")
    print("1. Add your training images to: ../datasets/secureeye/train/images/")
    print("2. Add training labels to: ../datasets/secureeye/train/labels/")
    print("3. Add validation images to: ../datasets/secureeye/val/images/")
    print("4. Add validation labels to: ../datasets/secureeye/val/labels/")
    print("5. Add test images to: ../datasets/secureeye/test/images/")
    print("6. Add test labels to: ../datasets/secureeye/test/labels/")
    print("\nTraining command:")
    print("python train.py --data data/secureeye.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 100")
    print("\nNote: Replace sample label files with your actual labeled data.")

if __name__ == "__main__":
    main() 
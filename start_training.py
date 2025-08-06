#!/usr/bin/env python3
"""
SecureEye Training Starter Script
This script helps you start training the SecureEye model with proper configuration.
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dataset_structure():
    """Check if the dataset structure exists"""
    base_path = Path('../datasets/secureeye')
    required_dirs = [
        base_path / 'train' / 'images',
        base_path / 'train' / 'labels',
        base_path / 'val' / 'images',
        base_path / 'val' / 'labels',
        base_path / 'test' / 'images',
        base_path / 'test' / 'labels'
    ]
    
    missing_dirs = []
    for dir_path in required_dirs:
        if not dir_path.exists():
            missing_dirs.append(str(dir_path))
    
    if missing_dirs:
        print("❌ Missing dataset directories:")
        for dir_path in missing_dirs:
            print(f"   - {dir_path}")
        print("\nRun 'python setup_dataset.py' first to create the dataset structure.")
        return False
    
    return True

def check_training_data():
    """Check if training data is available"""
    train_images = Path('../datasets/secureeye/train/images')
    train_labels = Path('../datasets/secureeye/train/labels')
    
    image_files = list(train_images.glob('*.jpg')) + list(train_images.glob('*.jpeg')) + list(train_images.glob('*.png'))
    label_files = list(train_labels.glob('*.txt'))
    
    print(f"📊 Dataset Status:")
    print(f"   Training images: {len(image_files)}")
    print(f"   Training labels: {len(label_files)}")
    
    if len(image_files) == 0:
        print("⚠️  No training images found!")
        print("   Add images to: ../datasets/secureeye/train/images/")
        return False
    
    if len(label_files) == 0:
        print("⚠️  No training labels found!")
        print("   Add labels to: ../datasets/secureeye/train/labels/")
        return False
    
    return True

def get_training_command(epochs=100, batch_size=16, img_size=640, weights='yolov5s.pt'):
    """Generate the training command"""
    cmd = [
        'python', 'train.py',
        '--data', 'data/secureeye.yaml',
        '--weights', weights,
        '--img', str(img_size),
        '--batch', str(batch_size),
        '--epochs', str(epochs),
        '--project', 'runs/train',
        '--name', 'secureeye_model',
        '--exist-ok'
    ]
    return cmd

def main():
    """Main function"""
    print("=" * 60)
    print("SecureEye Training Starter")
    print("=" * 60)
    
    # Check dataset structure
    if not check_dataset_structure():
        return
    
    # Check training data
    if not check_training_data():
        print("\n💡 To get started with sample data:")
        print("   1. Download some surveillance images")
        print("   2. Annotate them using tools like LabelImg or CVAT")
        print("   3. Place images in train/images/ and labels in train/labels/")
        print("   4. Run this script again")
        return
    
    print("\n✅ Dataset ready for training!")
    
    # Training options
    print("\n🎯 Training Options:")
    print("1. Quick training (50 epochs, small batch)")
    print("2. Standard training (100 epochs, medium batch)")
    print("3. Full training (200 epochs, large batch)")
    print("4. Custom training")
    
    choice = input("\nSelect training option (1-4): ").strip()
    
    if choice == '1':
        epochs, batch_size = 50, 8
        print(f"\n🚀 Starting quick training: {epochs} epochs, batch size {batch_size}")
    elif choice == '2':
        epochs, batch_size = 100, 16
        print(f"\n🚀 Starting standard training: {epochs} epochs, batch size {batch_size}")
    elif choice == '3':
        epochs, batch_size = 200, 32
        print(f"\n🚀 Starting full training: {epochs} epochs, batch size {batch_size}")
    elif choice == '4':
        try:
            epochs = int(input("Enter number of epochs: "))
            batch_size = int(input("Enter batch size: "))
            print(f"\n🚀 Starting custom training: {epochs} epochs, batch size {batch_size}")
        except ValueError:
            print("❌ Invalid input. Using default values.")
            epochs, batch_size = 100, 16
    else:
        print("❌ Invalid choice. Using default values.")
        epochs, batch_size = 100, 16
    
    # Check if weights file exists
    weights = 'yolov5s.pt'
    if not Path(weights).exists():
        print(f"⚠️  Weights file '{weights}' not found.")
        print("   Downloading pretrained weights...")
        try:
            import torch
            torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            print("✅ Weights downloaded successfully!")
        except Exception as e:
            print(f"❌ Failed to download weights: {e}")
            print("   You can download manually from: https://github.com/ultralytics/yolov5/releases")
            return
    
    # Generate and display command
    cmd = get_training_command(epochs, batch_size, 640, weights)
    print(f"\n📝 Training command:")
    print(" ".join(cmd))
    
    # Ask for confirmation
    confirm = input("\nStart training? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        print("\n🚀 Starting training...")
        print("   Press Ctrl+C to stop training")
        print("   Check runs/train/secureeye_model/ for results")
        print("-" * 60)
        
        try:
            subprocess.run(cmd, check=True)
        except KeyboardInterrupt:
            print("\n⏹️  Training stopped by user")
        except subprocess.CalledProcessError as e:
            print(f"\n❌ Training failed with error: {e}")
    else:
        print("❌ Training cancelled")

if __name__ == "__main__":
    main() 
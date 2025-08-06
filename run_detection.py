#!/usr/bin/env python3
"""
Simple wrapper script to run YOLOv5 live detection with different configurations.
This script provides easy-to-use functions for different detection scenarios.
"""

import subprocess
import sys
import os
from pathlib import Path

def run_basic_detection():
    """Run basic live detection with default settings."""
    print("Starting basic live detection...")
    cmd = [sys.executable, "detect_live.py", "--source", "0"]
    subprocess.run(cmd)

def run_custom_detection(weights="yolov5s.pt", source="0", conf_thres=0.25):
    """Run detection with custom parameters."""
    print(f"Starting detection with weights: {weights}, source: {source}, confidence: {conf_thres}")
    cmd = [
        sys.executable, "detect_live.py",
        "--weights", weights,
        "--source", source,
        "--conf-thres", str(conf_thres)
    ]
    subprocess.run(cmd)

def run_person_detection():
    """Run detection focused on person detection only."""
    print("Starting person-only detection...")
    cmd = [
        sys.executable, "detect_live.py",
        "--source", "0",
        "--classes", "0",  # Person class in COCO dataset
        "--conf-thres", "0.5"
    ]
    subprocess.run(cmd)

def run_high_confidence_detection():
    """Run detection with high confidence threshold for more accurate results."""
    print("Starting high-confidence detection...")
    cmd = [
        sys.executable, "detect_live.py",
        "--source", "0",
        "--conf-thres", "0.7",
        "--iou-thres", "0.5"
    ]
    subprocess.run(cmd)

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = ['torch', 'torchvision', 'opencv-python', 'numpy', 'ultralytics']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"Missing packages: {', '.join(missing_packages)}")
        print("Install them with: pip install -r detection_requirements.txt")
        return False
    return True

def main():
    """Main function with menu options."""
    print("YOLOv5 Live Detection Runner")
    print("=" * 30)
    
    # Check dependencies
    if not check_dependencies():
        return
    
    while True:
        print("\nSelect an option:")
        print("1. Basic detection (default settings)")
        print("2. Person-only detection")
        print("3. High-confidence detection")
        print("4. Custom detection")
        print("5. Exit")
        
        choice = input("\nEnter your choice (1-5): ").strip()
        
        if choice == "1":
            run_basic_detection()
        elif choice == "2":
            run_person_detection()
        elif choice == "3":
            run_high_confidence_detection()
        elif choice == "4":
            weights = input("Enter weights file (default: yolov5s.pt): ").strip() or "yolov5s.pt"
            source = input("Enter camera source (default: 0): ").strip() or "0"
            conf = input("Enter confidence threshold (default: 0.25): ").strip() or "0.25"
            try:
                conf_float = float(conf)
                run_custom_detection(weights, source, conf_float)
            except ValueError:
                print("Invalid confidence threshold. Using default 0.25")
                run_custom_detection(weights, source, 0.25)
        elif choice == "5":
            print("Goodbye!")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 
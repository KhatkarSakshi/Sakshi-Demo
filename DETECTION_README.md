# YOLOv5 Live Detection System

This directory contains an improved and robust implementation of YOLOv5 live object detection with comprehensive error handling and configuration options.

## Files Overview

- `detect_live.py` - Main detection script with full error handling
- `run_detection.py` - Simple wrapper script with menu options
- `detection_requirements.txt` - Minimal requirements for detection
- `DETECTION_README.md` - This documentation file

## Installation

1. **Install Dependencies:**
   ```bash
   pip install -r detection_requirements.txt
   ```

2. **Verify Installation:**
   ```bash
   python run_detection.py
   ```

## Usage

### Method 1: Using the Wrapper Script (Recommended for Beginners)

```bash
python run_detection.py
```

This will show a menu with options:
- Basic detection (default settings)
- Person-only detection
- High-confidence detection
- Custom detection

### Method 2: Direct Command Line Usage

```bash
# Basic detection with webcam
python detect_live.py --source 0

# Person detection only
python detect_live.py --source 0 --classes 0 --conf-thres 0.5

# High confidence detection
python detect_live.py --source 0 --conf-thres 0.7

# Custom model weights
python detect_live.py --weights yolov5m.pt --source 0

# Use GPU (if available)
python detect_live.py --source 0 --device 0
```

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--weights` | `yolov5s.pt` | Model weights path |
| `--source` | `0` | Camera source (0 for webcam) |
| `--img-size` | `640` | Inference size (pixels) |
| `--conf-thres` | `0.25` | Confidence threshold |
| `--iou-thres` | `0.45` | NMS IoU threshold |
| `--device` | `''` | CUDA device (0,1,2,3) or 'cpu' |
| `--classes` | `None` | Filter by class (e.g., 0 for person) |
| `--half` | `False` | Use FP16 half-precision inference |
| `--hide-labels` | `False` | Hide labels on detection boxes |
| `--hide-conf` | `False` | Hide confidence scores |

## Key Features

### 1. **Robust Error Handling**
- Graceful handling of camera connection failures
- Model loading error recovery
- Frame processing error handling
- Keyboard interrupt handling

### 2. **Performance Optimization**
- Model warmup for faster inference
- Optional GPU acceleration
- Half-precision inference support
- Efficient image preprocessing

### 3. **User-Friendly Controls**
- Press `q` to quit detection
- Press `s` to save current frame
- Real-time performance metrics
- Detection count display

### 4. **Flexible Configuration**
- Multiple camera source support
- Custom confidence thresholds
- Class-specific detection
- Various model weights support

## Troubleshooting

### Common Issues

1. **Camera Not Found:**
   ```bash
   # Try different camera indices
   python detect_live.py --source 1
   python detect_live.py --source 2
   ```

2. **CUDA Out of Memory:**
   ```bash
   # Use CPU instead
   python detect_live.py --source 0 --device cpu
   
   # Or use smaller model
   python detect_live.py --weights yolov5n.pt --source 0
   ```

3. **Slow Performance:**
   ```bash
   # Reduce image size
   python detect_live.py --source 0 --img-size 320
   
   # Increase confidence threshold
   python detect_live.py --source 0 --conf-thres 0.5
   ```

4. **Import Errors:**
   ```bash
   # Make sure you're in the YOLOv5 directory
   cd yolov5
   python detect_live.py --source 0
   ```

### Performance Tips

1. **For Real-time Applications:**
   - Use `yolov5n.pt` or `yolov5s.pt` models
   - Set `--img-size 320` for faster processing
   - Use GPU if available: `--device 0`

2. **For High Accuracy:**
   - Use `yolov5l.pt` or `yolov5x.pt` models
   - Set `--img-size 640` or higher
   - Use `--conf-thres 0.5` or higher

3. **For Specific Object Detection:**
   - Use `--classes` to filter specific objects
   - Adjust confidence threshold based on needs

## Model Weights

Available pre-trained models:
- `yolov5n.pt` - Nano (fastest, smallest)
- `yolov5s.pt` - Small (balanced)
- `yolov5m.pt` - Medium (good accuracy)
- `yolov5l.pt` - Large (high accuracy)
- `yolov5x.pt` - Extra Large (best accuracy)

## COCO Dataset Classes

Common class indices:
- `0` - person
- `1` - bicycle
- `2` - car
- `3` - motorcycle
- `5` - bus
- `7` - truck
- `16` - dog
- `17` - horse
- `39` - bottle
- `67` - cell phone

## Examples

```bash
# Detect only people with high confidence
python detect_live.py --source 0 --classes 0 --conf-thres 0.7

# Detect vehicles only
python detect_live.py --source 0 --classes 1 2 3 5 7 --conf-thres 0.5

# Use GPU with large model for maximum accuracy
python detect_live.py --weights yolov5l.pt --source 0 --device 0 --img-size 640

# Fast detection with nano model
python detect_live.py --weights yolov5n.pt --source 0 --img-size 320 --conf-thres 0.3
```

## Support

If you encounter issues:
1. Check that all dependencies are installed
2. Verify you're running from the YOLOv5 directory
3. Ensure your camera is working and accessible
4. Try different camera sources if needed
5. Check GPU drivers if using CUDA 
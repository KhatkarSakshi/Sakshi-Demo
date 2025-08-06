# SecureEye Training Dataset Guide

This guide explains how to set up and use the training dataset for the SecureEye surveillance system.

## Overview

SecureEye is an intelligent surveillance system that uses YOLOv5 to detect various security events including:
- Fire and smoke detection
- Weapon detection
- Intrusion detection
- Violence detection
- Crowd monitoring
- Suspicious activities
- And more...

## Quick Start

1. **Set up the dataset structure:**
   ```bash
   cd yolov5
   python setup_dataset.py
   ```

2. **Add your training data:**
   - Place images in `../datasets/secureeye/train/images/`
   - Place corresponding labels in `../datasets/secureeye/train/labels/`
   - Repeat for validation and test sets

3. **Start training:**
   ```bash
   python train.py --data data/secureeye.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 100
   ```

## Dataset Structure

```
datasets/
└── secureeye/
    ├── train/
    │   ├── images/     # Training images
    │   └── labels/     # Training labels (YOLO format)
    ├── val/
    │   ├── images/     # Validation images
    │   └── labels/     # Validation labels
    └── test/
        ├── images/     # Test images
        └── labels/     # Test labels
```

## Classes

The SecureEye model detects 15 different classes:

| Class ID | Class Name | Description |
|----------|------------|-------------|
| 0 | person | Individual person detection |
| 1 | fire | Fire/flame detection |
| 2 | smoke | Smoke detection |
| 3 | weapon | Weapon/firearm detection |
| 4 | vehicle | Vehicle detection |
| 5 | crowd | Crowd/gathering detection |
| 6 | intrusion | Unauthorized entry detection |
| 7 | violence | Violent behavior detection |
| 8 | running | Running person detection |
| 9 | falling | Person falling detection |
| 10 | suspicious_activity | General suspicious behavior |
| 11 | unauthorized_access | Unauthorized access detection |
| 12 | vandalism | Property damage detection |
| 13 | theft | Theft/stealing detection |
| 14 | loitering | Loitering behavior detection |

## Label Format

Labels must be in YOLO format: `class_id x_center y_center width height`

Example:
```
0 0.5 0.5 0.2 0.4    # Person at center
1 0.3 0.7 0.4 0.3    # Fire in bottom-left
3 0.8 0.2 0.1 0.2    # Weapon in top-right
```

## Data Collection Guidelines

### Image Requirements
- **Resolution**: Minimum 640x640 pixels
- **Format**: JPG, JPEG, or PNG
- **Quality**: High-quality, clear images
- **Lighting**: Various conditions (day, night, low-light)
- **Weather**: Different weather conditions

### Camera Angles
- CCTV-style overhead views
- Security camera angles
- Various distances and perspectives
- Different installation heights

### Scenarios to Capture
- **Normal activities** (baseline for comparison)
- **Fire and smoke incidents**
- **Security breaches**
- **Crowd gatherings**
- **Suspicious behaviors**
- **Emergency situations**
- **Weapon incidents**
- **Violence scenarios**

### Annotation Guidelines
- Use accurate bounding boxes
- Include all visible instances
- Label occluded objects when possible
- Ensure consistent labeling across similar scenarios
- Use the correct class IDs

## Dataset Split Recommendations

- **Training**: 70% of total data
- **Validation**: 20% of total data  
- **Test**: 10% of total data

## Class Distribution Guidelines

For balanced training, aim for this distribution:

| Class | Target Percentage | Reasoning |
|-------|------------------|-----------|
| person | 30% | Most common in surveillance |
| fire | 5% | Critical for safety |
| smoke | 5% | Fire precursor |
| weapon | 3% | Security critical |
| vehicle | 15% | Common in surveillance |
| crowd | 10% | Important for monitoring |
| intrusion | 8% | Security event |
| violence | 5% | Critical security event |
| running | 8% | Suspicious activity |
| falling | 3% | Safety concern |
| suspicious_activity | 5% | General suspicious behavior |
| unauthorized_access | 3% | Security breach |
| vandalism | 2% | Property damage |
| theft | 2% | Criminal activity |
| loitering | 5% | Suspicious behavior |

## Training Commands

### Basic Training
```bash
python train.py --data data/secureeye.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 100
```

### Training with Custom Hyperparameters
```bash
python train.py --data data/secureeye.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 100 --hyp data/hyps/hyp.scratch-low.yaml
```

### Multi-GPU Training
```bash
python -m torch.distributed.run --nproc_per_node 4 --master_port 1 train.py --data data/secureeye.yaml --weights yolov5s.pt --img 640 --batch 16 --epochs 100 --device 0,1,2,3
```

### Resume Training
```bash
python train.py --data data/secureeye.yaml --weights runs/train/secureeye_model/weights/last.pt --img 640 --batch 16 --epochs 100 --resume
```

## Validation and Testing

### Validate Model
```bash
python val.py --data data/secureeye.yaml --weights runs/train/secureeye_model/weights/best.pt --img 640
```

### Test on Images
```bash
python detect.py --weights runs/train/secureeye_model/weights/best.pt --source path/to/test/images --img 640 --conf 0.25
```

### Live Detection
```bash
python detect_live.py --weights runs/train/secureeye_model/weights/best.pt --source 0 --img 640 --conf 0.25
```

## Data Augmentation

The training uses these augmentation settings optimized for surveillance:

- **HSV augmentation**: Color variations for different lighting
- **Translation**: Position variations
- **Scale**: Size variations
- **Flip**: Horizontal flipping (50% probability)
- **Mosaic**: Image mosaic augmentation (100% probability)

## Model Performance

Expected performance metrics:
- **mAP@0.5**: > 0.8 for critical classes (fire, weapon, violence)
- **mAP@0.5:0.95**: > 0.6 overall
- **Precision**: > 0.85 for security-critical classes
- **Recall**: > 0.8 for security-critical classes

## Troubleshooting

### Common Issues

1. **Low mAP scores**: Check class balance and annotation quality
2. **Overfitting**: Reduce model complexity or increase data
3. **Poor detection**: Ensure sufficient training data for each class
4. **Memory issues**: Reduce batch size or image size

### Tips for Better Results

1. **Data Quality**: Use high-quality, well-annotated images
2. **Class Balance**: Ensure all classes have sufficient samples
3. **Variety**: Include diverse scenarios and conditions
4. **Validation**: Use a representative validation set
5. **Iteration**: Train multiple models and ensemble results

## Support

For questions or issues with the dataset setup:
1. Check the YOLOv5 documentation
2. Review the sample files created by `setup_dataset.py`
3. Ensure your data follows the YOLO format
4. Verify class IDs match the configuration

## License

This dataset configuration is part of the SecureEye project and follows the same license as YOLOv5 (AGPL-3.0). 
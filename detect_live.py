##user!/usr/bin/env python3#
"""
YOLOv5 Live Object Detection Script
A robust implementation for real-time object detection using webcam feed.
"""

import sys
import time
import argparse
import cv2
import torch
import numpy as np
from pathlib import Path

# Add YOLOv5 to path
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
    

try:
    from models.common import DetectMultiBackend
    from utils.general import (check_img_size, non_max_suppression, scale_boxes, 
                              check_imshow, xyxy2xywh, increment_path)
    from utils.plots import Annotator, colors
    from utils.torch_utils import select_device, time_sync
    from utils.augmentations import letterbox
except ImportError as e:
    print(f"Error importing YOLOv5 modules: {e}")
    print("Make sure you're running this script from the YOLOv5 directory")
    sys.exit(1)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='YOLOv5 Live Detection')
    parser.add_argument('--weights', type=str, default='yolov5s.pt', 
                       help='model weights path')
    parser.add_argument('--source', type=str, default='0', 
                       help='camera source (0 for webcam)')
    parser.add_argument('--img-size', type=int, default=640, 
                       help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, 
                       help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, 
                       help='NMS IoU threshold')
    parser.add_argument('--max-det', type=int, default=1000, 
                       help='maximum detections per image')
    parser.add_argument('--device', default='', 
                       help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--classes', nargs='+', type=int, 
                       help='filter by class: --classes 0, or --classes 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', 
                       help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', 
                       help='augmented inference')
    parser.add_argument('--visualize', action='store_true', 
                       help='visualize features')
    parser.add_argument('--update', action='store_true', 
                       help='update all models')
    parser.add_argument('--project', default='runs/detect', 
                       help='save results to project/name')
    parser.add_argument('--name', default='exp', 
                       help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', 
                       help='existing project/name ok, do not increment')
    parser.add_argument('--line-thickness', default=3, type=int, 
                       help='bounding box thickness (pixels)')
    parser.add_argument('--hide-labels', default=False, action='store_true', 
                       help='hide labels')
    parser.add_argument('--hide-conf', default=False, action='store_true', 
                       help='hide confidences')
    parser.add_argument('--half', action='store_true', 
                       help='use FP16 half-precision inference')
    parser.add_argument('--dnn', action='store_true', 
                       help='use OpenCV DNN for ONNX inference')
    return parser.parse_args()


def load_model(weights, device, dnn=False, half=False):
    """Load YOLOv5 model."""
    try:
        model = DetectMultiBackend(weights, device=device, dnn=dnn, data=None, fp16=half)
        stride, names, pt = model.stride, model.names, model.pt
        imgsz = check_img_size((640, 640), s=stride)  # check image size
        return model, stride, names, pt, imgsz
    except Exception as e:
        print(f"Error loading model: {e}")
        return None, None, None, None, None


def preprocess_image(im, imgsz, stride, pt):
    """Preprocess image for inference."""
    try:
        # Padded resize
        im = letterbox(im, imgsz, stride=stride, auto=pt)[0]
        # Convert
        im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        im = np.ascontiguousarray(im)
        im = torch.from_numpy(im).to(device)
        im = im.half() if half else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        return im
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None


def run_inference(model, im, augment=False, visualize=False):
    """Run YOLOv5 inference on image."""
    try:
        # Inference
        pred = model(im, augment=augment, visualize=visualize)
        return pred
    except Exception as e:
        print(f"Error during inference: {e}")
        return None


def post_process(pred, im, im0s, conf_thres, iou_thres, classes, agnostic_nms, max_det):
    """Post-process detection results."""
    try:
        # NMS
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        return pred
    except Exception as e:
        print(f"Error in post-processing: {e}")
        return None


def draw_detections(pred, im0, annotator, names):
    """Draw detection boxes on image."""
    try:
        if len(pred):
            # Rescale boxes from img_size to im0 size
            pred[0][:, :4] = scale_boxes(im.shape[2:], pred[0][:, :4], im0.shape).round()

            # Print results
            for c in pred[0][:, -1].unique():
                n = (pred[0][:, -1] == c).sum()  # detections per class
                print(f"Detected {n} {names[int(c)]}{'s' * (n > 1)}")

            # Write results
            for *xyxy, conf, cls in reversed(pred[0]):
                c = int(cls)  # integer class
                label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                annotator.box_label(xyxy, label, color=colors(c, True))
        
        return im0
    except Exception as e:
        print(f"Error drawing detections: {e}")
        return im0


def main():
    """Main function for live detection."""
    global device, half, hide_labels, hide_conf
    
    # Parse arguments
    opt = parse_arguments()
    
    # Initialize
    device = select_device(opt.device)
    half = opt.half and device.type != 'cpu'  # half precision only supported on CUDA
    hide_labels = opt.hide_labels
    hide_conf = opt.hide_conf
    
    # Load model
    print(f"Loading model from {opt.weights}...")
    model, stride, names, pt, imgsz = load_model(opt.weights, device, opt.dnn, half)
    if model is None:
        print("Failed to load model. Exiting.")
        return
    
    # Load data
    source = str(opt.source)
    is_file = Path(source).suffix[1:] in ('jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif', 'webp')
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    
    # Dataloader
    if webcam:
        print(f"Starting webcam from source: {source}")
        cap = cv2.VideoCapture(int(source) if source.isnumeric() else source)
        if not cap.isOpened():
            print(f"Error: Could not open camera source {source}")
            return
    else:
        print(f"Processing file: {source}")
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video file {source}")
            return
    
    # Run inference
    model.warmup(imgsz=(1 if pt else 1, 3, *imgsz))  # warmup
    dt, seen = [0.0, 0.0, 0.0], 0
    
    print("Starting live detection. Press 'q' to quit, 's' to save frame.")
    
    try:
        while True:
            # Read frame
            ret, im0 = cap.read()
            if not ret:
                print("Failed to read frame. Exiting.")
                break
            
            # Preprocess
            im = preprocess_image(im0, imgsz, stride, pt)
            if im is None:
                continue
            
            # Inference
            t1 = time_sync()
            pred = run_inference(model, im, opt.augment, opt.visualize)
            if pred is None:
                continue
            t2 = time_sync()
            
            # Post-process
            pred = post_process(pred, im, im0, opt.conf_thres, opt.iou_thres, 
                              opt.classes, opt.agnostic_nms, opt.max_det)
            if pred is None:
                continue
            t3 = time_sync()
            
            # Process detections
            for i, det in enumerate(pred):  # per image
                seen += 1
                im0_copy = im0.copy()
                
                annotator = Annotator(im0_copy, line_width=opt.line_thickness, example=str(names))
                
                if len(det):
                    # Rescale boxes from img_size to im0 size
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0_copy.shape).round()
                    
                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        print(f"Detected {n} {names[int(c)]}{'s' * (n > 1)}")
                    
                    # Write results
                    for *xyxy, conf, cls in reversed(det):
                        c = int(cls)  # integer class
                        label = None if hide_labels else (names[c] if hide_conf else f'{names[c]} {conf:.2f}')
                        annotator.box_label(xyxy, label, color=colors(c, True))
                
                # Stream results
                cv2.imshow('YOLOv5 Live Detection', im0_copy)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    print("Quit requested by user.")
                    break
                elif key == ord('s'):
                    # Save current frame
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    filename = f"detection_{timestamp}.jpg"
                    cv2.imwrite(filename, im0_copy)
                    print(f"Frame saved as {filename}")
            
            # Print time (inference-only)
            print(f'Inference time: {(t2 - t1) * 1E3:.1f}ms, NMS time: {(t3 - t2) * 1E3:.1f}ms')
    
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    except Exception as e:
        print(f"Error during detection: {e}")
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Detection stopped.")


if __name__ == "__main__":
    main()
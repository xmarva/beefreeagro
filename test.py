import os
import sys
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO
from tqdm import tqdm

def draw_boxes(image, boxes, classes, scores, class_names):
    colors = {
        'happy': (0, 255, 0),    # Green
        'sad': (0, 0, 255),      # Red
        'dead': (255, 0, 0)      # Blue
    }
    
    img_copy = image.copy()
    
    for box, cls_idx, score in zip(boxes, classes, scores):
        x1, y1, x2, y2 = box
        cls_name = class_names[int(cls_idx)]
        color = colors[cls_name]
        
        cv2.rectangle(img_copy, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        text = f"{cls_name} {score:.2f}"
        font_scale = 0.5
        font_thickness = 1
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
        
        text_x = int(x1)
        text_y = int(y1) - 5 if int(y1) - 5 > text_size[1] else int(y1) + text_size[1] + 5
        
        cv2.rectangle(
            img_copy, 
            (text_x, text_y - text_size[1] - 5), 
            (text_x + text_size[0], text_y + 5), 
            color, 
            -1
        )
        
        cv2.putText(
            img_copy, 
            text, 
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 
            font_scale, 
            (255, 255, 255), 
            font_thickness
        )
    
    return img_copy

def run_detection(model_path, test_dir, result_dir, conf_threshold=0.25):
    print(f"Model path: {model_path}")
    print(f"Test directory: {test_dir}")
    print(f"Result directory: {result_dir}")
    print(f"Confidence threshold: {conf_threshold}")

    os.makedirs(result_dir, exist_ok=True)
    
    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} not found")
        print(f"Looking for model in standard location: models/best.pt")
        model_path = "models/best.pt"
        if not os.path.exists(model_path):
            print(f"Error: Model file {model_path} not found")
            return

    model = YOLO(model_path)
    class_names = model.names
    
    image_files = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
        image_files.extend(list(Path(test_dir).glob(ext)))
    
    if not image_files:
        print(f"No image files found in {test_dir}")
        return
    
    print(f"Running detection on {len(image_files)} images...")
    
    for img_path in tqdm(image_files):
        img_name = img_path.name
        img = cv2.imread(str(img_path))
        
        if img is None:
            print(f"Warning: Could not read image {img_path}")
            continue
        
        results = model(img, conf=conf_threshold)
        
        boxes = []
        classes = []
        scores = []
        
        if len(results) > 0:
            result = results[0]
            for box in result.boxes:
                # Get box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                boxes.append([x1, y1, x2, y2])
                
                cls_idx = int(box.cls.cpu().numpy()[0])
                score = float(box.conf.cpu().numpy()[0])
                
                classes.append(cls_idx)
                scores.append(score)
        
        img_with_boxes = draw_boxes(img, boxes, classes, scores, class_names)
        
        cv2.imwrite(os.path.join(result_dir, img_name), img_with_boxes)
        
        print(f"{img_name} ", end="")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = [int(coord) for coord in box]
            cls_name = class_names[classes[i]]
            print(f"({x1},{y1})-({x2},{y2}) {cls_name} ", end="")
        print()

def main():
    parser = argparse.ArgumentParser(description='Test YOLO model for sticker detection')
    parser.add_argument('test_dir', type=str, help='Directory with test images')
    parser.add_argument('result_dir', type=str, help='Directory to save results')
    parser.add_argument('--model', type=str, default='models/best.pt', help='Path to trained model')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    
    if len(sys.argv) < 3:
        parser.print_help()
        sys.exit(1)
    
    args = parser.parse_args()
    
    run_detection(args.model, args.test_dir, args.result_dir, args.conf)

if __name__ == "__main__":
    main()
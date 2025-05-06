import os
import yaml
import shutil
import argparse
from pathlib import Path
import random
import cv2
import re
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def create_yolo_dataset(data_dir, annotations_file, output_dir, val_split=0.2):
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for d in [train_dir, val_dir]:
        os.makedirs(os.path.join(d, 'images'), exist_ok=True)
        os.makedirs(os.path.join(d, 'labels'), exist_ok=True)
    
    with open(annotations_file, 'r') as f:
        lines = f.readlines()
    
    image_names = []
    for line in lines:
        image_name = line.split(' ')[0]
        if image_name not in image_names:
            image_names.append(image_name)
    
    random.shuffle(image_names)
    val_count = max(1, int(len(image_names) * val_split))
    val_images = image_names[:val_count]
    train_images = image_names[val_count:]
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    class_map = {'happy': 0, 'sad': 1, 'dead': 2}
    
    for img_name in tqdm(image_names, desc="Processing dataset"):
        is_val = img_name in val_images
        dest_dir = val_dir if is_val else train_dir
        
        src_img_path = os.path.join(data_dir, img_name)
        dst_img_path = os.path.join(dest_dir, 'images', img_name)
        shutil.copy2(src_img_path, dst_img_path)
        
        img_annotations = [line for line in lines if line.startswith(img_name + ' ')]
        
        label_filename = os.path.splitext(img_name)[0] + '.txt'
        dst_label_path = os.path.join(dest_dir, 'labels', label_filename)
        
        img = cv2.imread(src_img_path)
        if img is None:
            print(f"Warning: Could not read image {src_img_path}")
            continue
            
        height, width = img.shape[:2]
        
        with open(dst_label_path, 'w') as label_file:
            for annotation in img_annotations:
                # Parse annotation (imgname (x1,y1)-(x2,y2) class)
                parts = annotation.strip().split(' ')
                img_name_ann = parts[0]
                
                annotation_text = ' '.join(parts[1:])
                
                import re
                box_pattern = r'\((\d+),(\d+)\)-\((\d+),(\d+)\)\s+(\w+)'
                matches = re.findall(box_pattern, annotation_text)
                
                for match in matches:
                    x1, y1, x2, y2, class_name = match
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Convert to YOLO format (center_x, center_y, width, height) - normalized
                    center_x = (x1 + x2) / 2 / width
                    center_y = (y1 + y2) / 2 / height
                    box_width = (x2 - x1) / width
                    box_height = (y2 - y1) / height
                    
                    if class_name not in class_map:
                        print(f"Warning: Unknown class '{class_name}' in {annotation}")
                        continue
                        
                    class_idx = class_map[class_name]
                    
                    # Write to label file
                    label_file.write(f"{class_idx} {center_x} {center_y} {box_width} {box_height}\n")
    
    dataset_yaml = {
        'path': os.path.abspath(output_dir),
        'train': os.path.join('train', 'images'),
        'val': os.path.join('val', 'images'),
        'names': {0: 'happy', 1: 'sad', 2: 'dead'},
        'nc': 3
    }
    
    with open(os.path.join(output_dir, 'dataset.yaml'), 'w') as f:
        yaml.dump(dataset_yaml, f)
    
    return os.path.join(output_dir, 'dataset.yaml')

def train_model(config_path, model_size='n', epochs=100, batch_size=16, device='auto'):
    output_dir = os.path.join('models')
    os.makedirs(output_dir, exist_ok=True)
    
    model = YOLO(f'yolov8{model_size}.pt')
    
    results = model.train(
        data=config_path,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        patience=20,
        device=device,
        project=output_dir,
        name='stickers_detector',
        verbose=True
    )
    
    runs_dir = Path(output_dir) / 'stickers_detector'
    
    best_model = runs_dir / 'weights' / 'best.pt'
    if best_model.exists():
        target_path = Path(output_dir) / 'best.pt'
        shutil.copy2(best_model, target_path)
        print(f"Best model saved to {target_path}")
        return str(target_path)
    else:
        print(f"Warning: Could not find best model at {best_model}")
        # Fallback to last model
        last_model = runs_dir / 'weights' / 'last.pt'
        if last_model.exists():
            target_path = Path(output_dir) / 'best.pt'
            shutil.copy2(last_model, target_path)
            print(f"Last model saved to {target_path} (as best.pt)")
            return str(target_path)
    
    print(f"Models should be available in {runs_dir}/weights/")
    return None

def main():
    parser = argparse.ArgumentParser(description='Train YOLO model for sticker detection')
    parser.add_argument('--data_dir', type=str, default='data/synth/imgs', 
                        help='Directory with images')
    parser.add_argument('--annotations', type=str, default='data/synth/annotations.txt', 
                        help='Path to annotations file')
    parser.add_argument('--model_size', type=str, default='n', choices=['n', 's', 'm', 'l', 'x'],
                        help='YOLOv8 model size')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--device', type=str, default='auto', choices=['cpu', 'cuda', 'auto'],
                        help='Device to use for training')
    args = parser.parse_args()
    
    formatted_data_dir = os.path.join('data', 'formatted')
    config_path = create_yolo_dataset(args.data_dir, args.annotations, formatted_data_dir)
    
    model_path = train_model(
        config_path=config_path,
        model_size=args.model_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=args.device
    )
    
    if model_path:
        print(f"Training completed. Model saved at {model_path}")
    else:
        print("Training failed or no model was saved.")

if __name__ == "__main__":
    main()
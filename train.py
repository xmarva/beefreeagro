import os
import yaml
import shutil
import argparse
from pathlib import Path
import random
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

def create_yolo_dataset(data_dir, annotations_file, output_dir, val_split=0.2):
    """
    Create a YOLO-compatible dataset from annotations and images.
    
    Args:
        data_dir: Directory containing images
        annotations_file: Path to annotations file
        output_dir: Directory to save formatted dataset
        val_split: Fraction of data to use for validation
    """
    # Create directories
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, 'train')
    val_dir = os.path.join(output_dir, 'val')
    
    for d in [train_dir, val_dir]:
        os.makedirs(os.path.join(d, 'images'), exist_ok=True)
        os.makedirs(os.path.join(d, 'labels'), exist_ok=True)
    
    # Read annotations
    with open(annotations_file, 'r') as f:
        lines = f.readlines()
    
    # Get unique image names
    image_names = []
    for line in lines:
        image_name = line.split(' ')[0]
        if image_name not in image_names:
            image_names.append(image_name)
    
    # Shuffle and split
    random.shuffle(image_names)
    val_count = max(1, int(len(image_names) * val_split))
    val_images = image_names[:val_count]
    train_images = image_names[val_count:]
    
    print(f"Training images: {len(train_images)}")
    print(f"Validation images: {len(val_images)}")
    
    # Process annotations and copy images
    class_map = {'happy': 0, 'sad': 1, 'dead': 2}
    
    for img_name in tqdm(image_names, desc="Processing dataset"):
        # Determine destination
        is_val = img_name in val_images
        dest_dir = val_dir if is_val else train_dir
        
        # Copy image
        src_img_path = os.path.join(data_dir, img_name)
        dst_img_path = os.path.join(dest_dir, 'images', img_name)
        shutil.copy2(src_img_path, dst_img_path)
        
        # Get all annotations for this image
        img_annotations = [line for line in lines if line.startswith(img_name + ' ')]
        
        # Create label file
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
                
                # Extract all bounding boxes and classes
                i = 1
                while i < len(parts):
                    if parts[i].startswith('('):
                        box_str = parts[i] + ' ' + parts[i+1]
                        class_name = parts[i+2]
                        i += 3
                        
                        # Extract coordinates
                        coords = box_str.replace('(', '').replace(')', '').replace('-', ' ').split()
                        if len(coords) != 4:
                            print(f"Warning: Invalid coordinates in {annotation}")
                            continue
                            
                        x1, y1, x2, y2 = map(int, coords)
                        
                        # Convert to YOLO format (center_x, center_y, width, height) - normalized
                        center_x = (x1 + x2) / 2 / width
                        center_y = (y1 + y2) / 2 / height
                        box_width = (x2 - x1) / width
                        box_height = (y2 - y1) / height
                        
                        # Class index
                        class_idx = class_map[class_name]
                        
                        # Write to label file
                        label_file.write(f"{class_idx} {center_x} {center_y} {box_width} {box_height}\n")
                    else:
                        i += 1
    
    # Create YAML config file
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
    """
    Train YOLO model on the dataset.
    
    Args:
        config_path: Path to dataset YAML
        model_size: YOLO model size (n, s, m, l, x)
        epochs: Number of training epochs
        batch_size: Batch size
        device: Device to use ('cpu', 'cuda', or 'auto')
    """
    # Create output directory
    output_dir = os.path.join('models')
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize model
    model = YOLO(f'yolov8{model_size}.pt')
    
    # Train model
    model.train(
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
    
    # Copy best model to a standardized location
    runs_dir = Path(output_dir) / 'stickers_detector'
    last_run = sorted(runs_dir.iterdir(), key=lambda x: os.path.getmtime(x))[-1] if runs_dir.exists() and any(runs_dir.iterdir()) else None
    
    if last_run:
        best_model = last_run / 'weights' / 'best.pt'
        if best_model.exists():
            target_path = Path(output_dir) / 'best.pt'
            shutil.copy2(best_model, target_path)
            print(f"Best model saved to {target_path}")
            return str(target_path)
    
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
    
    # Create dataset
    formatted_data_dir = os.path.join('data', 'formatted')
    config_path = create_yolo_dataset(args.data_dir, args.annotations, formatted_data_dir)
    
    # Train model
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
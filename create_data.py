#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import random
import argparse
import math
from pathlib import Path
from typing import List, Tuple, Dict
import shutil
from tqdm import tqdm

class SyntheticDataGenerator:
    def __init__(
        self,
        stickers_dir: str,
        backgrounds_dir: str,
        output_dir: str,
        annotations_file: str,
        num_images: int = 1000,
        max_stickers_per_image: int = 5,
        min_stickers_per_image: int = 1,
        overlap_ratio: float = 0.3,
        random_seed: int = 42
    ):
        """
        Initialization of synthetic data generator
        
        Args:
            stickers_dir: Path to directory with stickers
            backgrounds_dir: Path to directory with backgrounds
            output_dir: Path to directory for saving generated images
            annotations_file: File name for saving annotations
            num_images: Number of images to generate
            max_stickers_per_image: Maximum number of stickers on one image
            min_stickers_per_image: Minimum number of stickers on one image
            overlap_ratio: Maximum stickers overlap ratio
            random_seed: Seed for random number generator
        """
        self.stickers_dir = stickers_dir
        self.backgrounds_dir = backgrounds_dir
        self.output_dir = output_dir
        self.annotations_file = annotations_file
        self.num_images = num_images
        self.max_stickers_per_image = max_stickers_per_image
        self.min_stickers_per_image = min_stickers_per_image
        self.overlap_ratio = overlap_ratio
        
        # Setting seed for reproducibility
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        # Creating output directories if they don't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Loading all stickers and backgrounds
        self.stickers = self._load_stickers()
        self.backgrounds = self._load_backgrounds()
        
        print(f"Loaded {len(self.stickers)} stickers and {len(self.backgrounds)} backgrounds")

    def _load_stickers(self) -> List[Dict]:
        """Loading stickers from directory"""
        stickers = []
        
        for file in os.listdir(self.stickers_dir):
            if file.lower().endswith(('.png')):
                path = os.path.join(self.stickers_dir, file)
                # Loading image with alpha channel
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                
                if img is None:
                    print(f"Error loading sticker: {path}")
                    continue
                
                # If the image doesn't have an alpha channel, add it
                if img.shape[2] == 3:
                    alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                    img = np.concatenate((img, alpha), axis=2)
                
                # Get class name from file name (without extension)
                class_name = os.path.splitext(file)[0]
                
                stickers.append({
                    'name': class_name,
                    'image': img,
                    'path': path
                })
        
        return stickers

    def _load_backgrounds(self) -> List[np.ndarray]:
        """Loading backgrounds from directory"""
        backgrounds = []
        
        for file in os.listdir(self.backgrounds_dir):
            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                path = os.path.join(self.backgrounds_dir, file)
                img = cv2.imread(path)
                
                if img is None:
                    print(f"Error loading background: {path}")
                    continue
                
                backgrounds.append(img)
        
        return backgrounds

    def _transform_sticker(self, sticker: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply various transformations to a sticker
        
        Args:
            sticker: Original sticker image with alpha channel
            
        Returns:
            Tuple (transformed image, mask)
        """
        # Resize (scaling)
        scale = random.uniform(0.5, 1.5)
        new_height = int(sticker.shape[0] * scale)
        new_width = int(sticker.shape[1] * scale)
        resized = cv2.resize(sticker, (new_width, new_height))
        
        # Rotation
        angle = random.uniform(-30, 30)
        center = (new_width // 2, new_height // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Getting dimensions after rotation to avoid cropping
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(new_height * abs_sin + new_width * abs_cos)
        bound_h = int(new_height * abs_cos + new_width * abs_sin)
        
        # Adjusting rotation matrix
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]
        
        # Applying rotation
        rotated = cv2.warpAffine(resized, rotation_matrix, (bound_w, bound_h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0, 0))
        
        # Extracting mask from alpha channel
        mask = rotated[:, :, 3] > 128
        
        # Changing brightness and contrast
        hsv = cv2.cvtColor(rotated[:, :, :3], cv2.COLOR_BGR2HSV)
        # Random brightness adjustment (V)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        # Random saturation adjustment (S)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * random.uniform(0.7, 1.3), 0, 255).astype(np.uint8)
        
        # Back to BGR color space
        rotated_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Blur
        if random.random() < 0.3:
            blur_size = random.choice([3, 5, 7])
            rotated_color = cv2.GaussianBlur(rotated_color, (blur_size, blur_size), 0)
        
        # Combining color channels with alpha channel
        final = np.zeros((rotated.shape[0], rotated.shape[1], 4), dtype=np.uint8)
        final[:, :, :3] = rotated_color
        final[:, :, 3] = rotated[:, :, 3]
        
        return final, mask.astype(np.uint8) * 255

    def _calculate_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Calculate bounding box from mask
        
        Args:
            mask: Binary sticker mask
            
        Returns:
            Tuple (x1, y1, x2, y2) of rectangle coordinates
        """
        # Find all non-zero elements of the mask
        y_indices, x_indices = np.where(mask > 0)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            return (0, 0, 0, 0)
        
        # Find minimum and maximum coordinates
        x1 = np.min(x_indices)
        y1 = np.min(y_indices)
        x2 = np.max(x_indices)
        y2 = np.max(y_indices)
        
        return (x1, y1, x2, y2)

    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        """
        Calculate overlap ratio between two bboxes
        
        Args:
            bbox1: First bounding box (x1, y1, x2, y2)
            bbox2: Second bounding box (x1, y1, x2, y2)
            
        Returns:
            Ratio of intersection area to the smaller bbox area
        """
        # Check for intersection
        if (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
            bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]):
            return 0.0
        
        # Calculate intersection coordinates
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # Calculate intersection area
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate bbox areas
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Find minimum area
        min_area = min(bbox1_area, bbox2_area)
        
        if min_area == 0:
            return 0.0
            
        # Calculate ratio
        return intersection_area / min_area

    def _place_sticker(self, background: np.ndarray, sticker: Dict, 
                       placed_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Tuple[int, int, int, int], str]:
        """
        Place sticker on background considering already placed stickers
        
        Args:
            background: Background image
            sticker: Dictionary with sticker information
            placed_bboxes: List of already placed bboxes
            
        Returns:
            Tuple (updated background, new sticker bbox, class name)
        """
        max_attempts = 20
        for _ in range(max_attempts):
            # Transform sticker
            transformed_sticker, mask = self._transform_sticker(sticker['image'])
            
            # Check that sticker has sufficient area
            if np.sum(mask > 0) < 100:
                continue
                
            # Calculate bbox for transformed sticker
            bbox_sticker = self._calculate_bbox(mask)
            
            sticker_width = bbox_sticker[2] - bbox_sticker[0]
            sticker_height = bbox_sticker[3] - bbox_sticker[1]
            
            # Check that sticker is not too small or too large
            if sticker_width < 20 or sticker_height < 20:
                continue
            if sticker_width > background.shape[1] * 0.8 or sticker_height > background.shape[0] * 0.8:
                continue
            
            # Choose random position for placement
            max_x = background.shape[1] - sticker_width
            max_y = background.shape[0] - sticker_height
            
            if max_x <= 0 or max_y <= 0:
                continue
                
            x_pos = random.randint(0, max_x)
            y_pos = random.randint(0, max_y)
            
            # Calculate new bbox considering position
            new_bbox = (
                x_pos + bbox_sticker[0],
                y_pos + bbox_sticker[1],
                x_pos + bbox_sticker[2],
                y_pos + bbox_sticker[3]
            )
            
            # Check overlap with other stickers
            overlap_too_much = False
            for placed_bbox in placed_bboxes:
                overlap = self._calculate_overlap(new_bbox, placed_bbox)
                if overlap > self.overlap_ratio:
                    overlap_too_much = True
                    break
            
            if overlap_too_much:
                continue
            
            # Create ROI for sticker placement
            roi = background[
                y_pos:y_pos + transformed_sticker.shape[0], 
                x_pos:x_pos + transformed_sticker.shape[1]
            ]
            
            # Check ROI and sticker dimensions
            if roi.shape[0] < transformed_sticker.shape[0] or roi.shape[1] < transformed_sticker.shape[1]:
                continue
                
            # Create copy of ROI for sticker overlay
            roi_copy = roi.copy()
            
            # Overlay sticker considering alpha channel
            for i in range(3):  # BGR channels
                roi_copy[:, :, i] = (
                    roi[:, :, i] * (1 - transformed_sticker[:, :, 3] / 255.0) + 
                    transformed_sticker[:, :, i] * (transformed_sticker[:, :, 3] / 255.0)
                )
            
            # Update background
            background[
                y_pos:y_pos + transformed_sticker.shape[0], 
                x_pos:x_pos + transformed_sticker.shape[1]
            ] = roi_copy
            
            return background, new_bbox, sticker['name']
        
        # If couldn't place sticker after all attempts
        return background, None, None

    def generate_image_with_all_stickers(self, img_index: int, background_img: np.ndarray) -> str:
        """
        Generate one synthetic image with all stickers placed exactly once
        
        Args:
            img_index: Image index
            background_img: Background image to use
            
        Returns:
            String with annotations for the generated image
        """
        # Create a copy of the background
        background = background_img.copy()
        
        # List to store placed bboxes
        placed_bboxes = []
        
        # String for annotations
        annotations = []
        
        # Place each sticker exactly once
        for sticker in self.stickers:
            # Place sticker on the background
            background, bbox, class_name = self._place_sticker(background, sticker, placed_bboxes)
            
            # If sticker was successfully placed
            if bbox is not None:
                placed_bboxes.append(bbox)
                # Add annotation in format (x1,y1)-(x2,y2) class_name
                annotation = f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]}) {class_name}"
                annotations.append(annotation)
        
        if len(annotations) == 0:
            # If no stickers were placed, skip this image
            return ""
            
        # Save the image
        output_path = os.path.join(self.output_dir, f"{img_index}.jpg")
        cv2.imwrite(output_path, background)
        
        # Format annotation string
        annotation_line = f"{img_index}.jpg " + " ".join(annotations)
        
        return annotation_line

    def generate_dataset(self):
        """Generate the entire dataset"""
        # Create annotations file
        annotations_path = os.path.join(self.output_dir, self.annotations_file)
        
        with open(annotations_path, 'w', encoding='utf-8') as f:
            img_idx = 0
            # For each background image
            for bg_idx, background in enumerate(tqdm(self.backgrounds, desc="Generating images")):
                # Generate one image with all stickers placed once
                annotation = self.generate_image_with_all_stickers(img_idx, background)
                if annotation:
                    f.write(annotation + '\n')
                    img_idx += 1
        
        print(f"Dataset generated and saved to {self.output_dir}")
        print(f"Annotations saved to {annotations_path}")


def main():
    parser = argparse.ArgumentParser(description='Synthetic data generator for YOLO training')
    
    parser.add_argument('--stickers_dir', type=str, default='data/raw/stickers',
                        help='Path to directory with stickers')
    parser.add_argument('--backgrounds_dir', type=str, default='data/raw/backgrounds',
                        help='Path to directory with backgrounds')
    parser.add_argument('--output_dir', type=str, default='data/synth',
                        help='Path to directory for saving generated images')
    parser.add_argument('--annotations_file', type=str, default='annotations.txt',
                        help='File name for saving annotations')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of images to generate')
    parser.add_argument('--max_stickers', type=int, default=5,
                        help='Maximum number of stickers on one image')
    parser.add_argument('--min_stickers', type=int, default=1,
                        help='Minimum number of stickers on one image')
    parser.add_argument('--overlap_ratio', type=float, default=0.3,
                        help='Maximum stickers overlap ratio')
    parser.add_argument('--random_seed', type=int, default=42,
                        help='Seed for random number generator')
    
    args = parser.parse_args()
    
    # Check if directories exist
    if not os.path.exists(args.stickers_dir):
        print(f"Error: Stickers directory not found: {args.stickers_dir}")
        return
    
    if not os.path.exists(args.backgrounds_dir):
        print(f"Error: Backgrounds directory not found: {args.backgrounds_dir}")
        return
    
    # Create generator and start generation
    generator = SyntheticDataGenerator(
        stickers_dir=args.stickers_dir,
        backgrounds_dir=args.backgrounds_dir,
        output_dir=args.output_dir,
        annotations_file=args.annotations_file,
        num_images=args.num_images,
        max_stickers_per_image=args.max_stickers,
        min_stickers_per_image=args.min_stickers,
        overlap_ratio=args.overlap_ratio,
        random_seed=args.random_seed
    )
    
    generator.generate_dataset()


if __name__ == "__main__":
    main()
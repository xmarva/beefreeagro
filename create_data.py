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
            overlap_ratio: Maximum stickers overlap ratio
            random_seed: Seed for random number generator
        """
        self.stickers_dir = stickers_dir
        self.backgrounds_dir = backgrounds_dir
        self.output_dir = output_dir
        self.annotations_file = annotations_file
        self.num_images = num_images
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

    def _get_background_color_info(self, background: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Extract dominant color tone and brightness from background
        
        Args:
            background: Background image
            
        Returns:
            Tuple (dominant_color, brightness)
        """
        # Convert to HSV for better color analysis
        hsv_bg = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        
        # Calculate average HSV values
        avg_h = np.mean(hsv_bg[:, :, 0])
        avg_s = np.mean(hsv_bg[:, :, 1])
        avg_v = np.mean(hsv_bg[:, :, 2])
        
        # Calculate brightness (normalized)
        brightness = avg_v / 255.0
        
        # Return dominant color and brightness
        return np.array([avg_h, avg_s, avg_v]), brightness

    def _apply_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        """
        Apply perspective transformation to the image
        
        Args:
            image: Original image with alpha channel
            
        Returns:
            Perspective transformed image
        """
        height, width = image.shape[:2]
        
        # Generate perspective transformation parameters with normal distribution
        # The larger the values, the more extreme the transformation
        # Horizontal perspective (left-right skew)
        h_skew = np.random.normal(0, 0.15)
        # Vertical perspective (top-bottom skew)
        v_skew = np.random.normal(0, 0.15)
        
        # Limit skew to reasonable values
        h_skew = max(-0.3, min(0.3, h_skew))
        v_skew = max(-0.3, min(0.3, v_skew))
        
        # Original corners
        pts1 = np.float32([
            [0, 0],               # top-left
            [width, 0],           # top-right
            [0, height],          # bottom-left
            [width, height]       # bottom-right
        ])
        
        # New corners after perspective transformation
        pts2 = np.float32([
            [width * max(0, -h_skew), height * max(0, -v_skew)],                           # top-left
            [width * (1 + max(0, h_skew)), height * max(0, -v_skew)],                      # top-right
            [width * max(0, -h_skew), height * (1 + max(0, v_skew))],                      # bottom-left
            [width * (1 + max(0, h_skew)), height * (1 + max(0, v_skew))]                  # bottom-right
        ])
        
        # Calculate transformation matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        # Apply transformation
        transformed = cv2.warpPerspective(
            image, 
            M, 
            (int(width * 1.5), int(height * 1.5)),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        # Find non-zero alpha channel values to crop the result
        alpha = transformed[:, :, 3]
        coords = cv2.findNonZero(alpha)
        
        if coords is None or len(coords) == 0:
            return image  # Return original if transform failed
        
        # Crop to content
        x, y, w, h = cv2.boundingRect(coords)
        cropped = transformed[y:y+h, x:x+w]
        
        return cropped

    def _add_noise(self, image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        """
        Add noise to the image
        
        Args:
            image: Original image (BGR)
            noise_level: Strength of noise (0.0-1.0)
            
        Returns:
            Noisy image
        """
        # Only add noise to RGB channels, not alpha
        result = image.copy()
        
        # Generate noise
        noise = np.random.normal(0, noise_level * 255, result.shape[:3])
        
        # Add noise to color channels only
        result[:, :, :3] = np.clip(result[:, :, :3] + noise[:, :, :3], 0, 255).astype(np.uint8)
        
        return result

    def _transform_sticker(self, sticker: np.ndarray, background: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply various transformations to a sticker considering background properties
        
        Args:
            sticker: Original sticker image with alpha channel
            background: Background image to adapt to
            
        Returns:
            Tuple (transformed image, mask)
        """
        # Get background information
        bg_color_info, bg_brightness = self._get_background_color_info(background)
        
        # Calculate target sticker size (around 0.1-10% of image area with normal distribution)
        # Modified to have smaller stickers
        bg_area = background.shape[0] * background.shape[1]
        target_sticker_area_ratio = np.random.normal(0.03, 0.02)
        # Ensure the ratio is reasonable (between 0.1% and 10%)
        target_sticker_area_ratio = max(0.001, min(0.1, target_sticker_area_ratio))
        
        # Calculate scale factor to achieve target area
        original_sticker_area = sticker.shape[0] * sticker.shape[1]
        target_sticker_area = bg_area * target_sticker_area_ratio
        scale = math.sqrt(target_sticker_area / original_sticker_area)
        
        # Resize (scaling)
        new_height = int(sticker.shape[0] * scale)
        new_width = int(sticker.shape[1] * scale)
        resized = cv2.resize(sticker, (new_width, new_height))
        
        # Apply perspective transformation
        if random.random() < 0.8:  # 80% chance to apply perspective
            resized = self._apply_perspective_transform(resized)
            if resized.size == 0:  # Check if transformation failed
                # Fallback to original resize
                resized = cv2.resize(sticker, (new_width, new_height))
        
        # Rotation - Full 360 degrees with uniform distribution
        angle = random.uniform(0, 360)
        center = (resized.shape[1] // 2, resized.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Getting dimensions after rotation to avoid cropping
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(resized.shape[0] * abs_sin + resized.shape[1] * abs_cos)
        bound_h = int(resized.shape[0] * abs_cos + resized.shape[1] * abs_sin)
        
        # Adjusting rotation matrix
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]
        
        # Applying rotation
        rotated = cv2.warpAffine(resized, rotation_matrix, (bound_w, bound_h), 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(0, 0, 0, 0))
        
        # Extracting mask from alpha channel
        mask = rotated[:, :, 3] > 128
        
        # Changing brightness and contrast based on background
        hsv = cv2.cvtColor(rotated[:, :, :3], cv2.COLOR_BGR2HSV)
        
        # Enhanced brightness adaptation (more extreme adjustments)
        # Darker background -> darker stickers or brighter background -> brighter stickers
        brightness_factor = 0.6 + bg_brightness * 0.8  # Range: 0.6-1.4 (more extreme)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)
        
        # More aggressive hue shift towards background color
        if random.random() < 0.8:  # Increased probability (80%)
            # Calculate hue shift (more significant shift)
            bg_hue = bg_color_info[0]
            hue_shift = (bg_hue - np.mean(hsv[:, :, 0])) * 0.25  # Increased from 0.15 to 0.25
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179).astype(np.uint8)
        
        # More aggressive color tinting from background
        # Lower threshold to apply tinting more often
        if bg_color_info[1] > 50:  # Reduced threshold from 100 to 50
            # Create a stronger color tint overlay
            overlay = np.ones_like(hsv)
            overlay[:, :, 0] = bg_color_info[0]  # Hue from background
            overlay[:, :, 1] = min(100, bg_color_info[1] * 0.5)  # Increased saturation influence
            overlay[:, :, 2] = 255  # Full value
            
            # Convert overlay to BGR
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_HSV2BGR)
            
            # Convert sticker to BGR
            sticker_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            # Stronger blend (70% sticker, 30% tint) - more influence from background
            sticker_bgr = cv2.addWeighted(sticker_bgr, 0.7, overlay_bgr, 0.3, 0)
            
            # Back to HSV
            hsv = cv2.cvtColor(sticker_bgr, cv2.COLOR_BGR2HSV)
        
        # Back to BGR color space
        rotated_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # Enhanced shadows based on background brightness
        if bg_brightness < 0.7:  # Increased threshold from 0.6 to 0.7
            # Create stronger shadow effect
            shadow_strength = 0.3 + (0.7 - bg_brightness) * 0.6  # Stronger shadow (0.3-0.9)
            kernel_size = int(max(5, min(15, rotated.shape[0] * 0.08)))  # Larger kernel
            if kernel_size % 2 == 0:
                kernel_size += 1  # Ensure odd kernel size
            shadow_mask = cv2.GaussianBlur(rotated[:, :, 3].copy(), (kernel_size, kernel_size), 0)
            shadow = np.zeros_like(rotated_color)
            for i in range(3):
                rotated_color[:, :, i] = np.clip(
                    rotated_color[:, :, i] * (1 - shadow_strength * shadow_mask / 255.0), 0, 255
                ).astype(np.uint8)
        
        # Enhanced brightness extremes (either very bright or very dark)
        if random.random() < 0.3:  # 30% chance
            # Extreme bright or dark adjustment
            if random.random() < 0.5:  # Bright
                brightness_boost = random.uniform(1.2, 1.5)
                rotated_color = np.clip(rotated_color * brightness_boost, 0, 255).astype(np.uint8)
            else:  # Dark
                darkness_factor = random.uniform(0.5, 0.8)
                rotated_color = np.clip(rotated_color * darkness_factor, 0, 255).astype(np.uint8)
        
        # Increased blur probability and strength
        if random.random() < 0.4:  # 40% chance (up from 30%)
            blur_size = random.choice([3, 5, 7, 9])  # Added stronger blur option
            rotated_color = cv2.GaussianBlur(rotated_color, (blur_size, blur_size), 0)
        
        # Add noise to the sticker
        if random.random() < 0.5:  # 50% chance to add noise
            noise_level = random.uniform(0.05, 0.2)  # Noise strength
            rotated_color = self._add_noise(rotated_color, noise_level)
        
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
        max_attempts = 30  # Increased attempts for better placement chance
        for _ in range(max_attempts):
            # Transform sticker with background adaptation
            transformed_sticker, mask = self._transform_sticker(sticker['image'], background)
            
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

    def generate_image(self, img_index: int, background_img: np.ndarray) -> str:
        """
        Generate one synthetic image with exactly 3 stickers
        
        Args:
            img_index: Image index
            background_img: Background image to use
            
        Returns:
            String with annotations for the generated image
        """
        # Make sure we have at least 3 stickers
        if len(self.stickers) < 3:
            print("Error: Need at least 3 stickers to generate images")
            return ""
            
        # Create a copy of the background
        background = background_img.copy()
        
        # List to store placed bboxes and annotations
        placed_bboxes = []
        annotations = []
        
        # Track stickers we've already placed
        placed_sticker_indices = set()
        
        # Try to place exactly 3 stickers (all different)
        max_placement_attempts = 50  # Total attempts to place all stickers
        
        for attempt in range(max_placement_attempts):
            # If we've already placed all 3 stickers, we're done
            if len(placed_bboxes) == 3:
                break
                
            # Choose a sticker we haven't placed yet
            available_indices = [i for i in range(len(self.stickers)) if i not in placed_sticker_indices]
            
            # If we've tried all stickers and failed, start over with any sticker
            if not available_indices:
                available_indices = list(range(len(self.stickers)))
                
            sticker_idx = random.choice(available_indices)
            sticker = self.stickers[sticker_idx]
            
            # Try to place the sticker
            background, bbox, class_name = self._place_sticker(background, sticker, placed_bboxes)
            
            # If sticker was successfully placed
            if bbox is not None:
                placed_bboxes.append(bbox)
                placed_sticker_indices.add(sticker_idx)
                
                # Add annotation in format (x1,y1)-(x2,y2) class_name
                annotation = f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]}) {class_name}"
                annotations.append(annotation)
        
        # Only save images with exactly 3 stickers
        if len(annotations) == 3:
            # Save the image
            output_path = os.path.join(self.output_dir, f"{img_index}.jpg")
            cv2.imwrite(output_path, background)
            
            # Format annotation string
            annotation_line = f"{img_index}.jpg " + " ".join(annotations)
            return annotation_line
        else:
            # If we couldn't place all 3 stickers, return empty string
            return ""

    def generate_dataset(self):
        """Generate the entire dataset"""
        # Create annotations file
        annotations_path = os.path.join(self.output_dir, self.annotations_file)
        
        with open(annotations_path, 'w', encoding='utf-8') as f:
            img_idx = 0
            generated_count = 0
            
            # Keep generating until we reach num_images or run out of backgrounds
            pbar = tqdm(total=self.num_images, desc="Generating images")
            bg_idx = 0
            
            while generated_count < self.num_images and bg_idx < len(self.backgrounds):
                # Get background
                background = self.backgrounds[bg_idx]
                
                # Try to generate image with 3 stickers
                annotation = self.generate_image(img_idx, background)
                
                # If successful, save annotation
                if annotation:
                    f.write(annotation + '\n')
                    img_idx += 1
                    generated_count += 1
                    pbar.update(1)
                
                # Move to next background
                bg_idx += 1
                
                # If we've used all backgrounds, start over
                if bg_idx >= len(self.backgrounds) and generated_count < self.num_images:
                    print("Used all backgrounds, restarting from the beginning")
                    bg_idx = 0
                    # Shuffle backgrounds for variety
                    random.shuffle(self.backgrounds)
            
            pbar.close()
        
        print(f"Dataset generated with {generated_count} images and saved to {self.output_dir}")
        print(f"Annotations saved to {annotations_path}")
        
        # Validate annotations
        self._validate_annotations(annotations_path)
    
    def _validate_annotations(self, annotations_path: str):
        """
        Validate the created annotations
        
        Args:
            annotations_path: Path to annotations file
        """
        print("Validating annotations...")
        with open(annotations_path, 'r') as f:
            lines = f.readlines()
        
        valid_count = 0
        invalid_count = 0
        sticker_counts = []
        
        for line in lines:
            parts = line.strip().split()
            if len(parts) < 2:
                print(f"Warning: Invalid line format: {line.strip()}")
                invalid_count += 1
                continue
            
            # Extract image name
            img_name = parts[0]
            
            # Count annotations for this image
            annotation_parts = line.strip().split(' ')[1:]
            annotation_count = 0
            
            # Process annotation parts
            for part in annotation_parts:
                if ')' in part and '(' in part:
                    annotation_count += 1
            
            sticker_counts.append(annotation_count)
            
            # Check if the image has exactly 3 stickers
            if annotation_count != 3:
                print(f"Warning: Image {img_name} has {annotation_count} stickers, expected 3")
                invalid_count += 1
            else:
                valid_count += 1
        
        print(f"Validation complete: {valid_count} valid images, {invalid_count} invalid images")
        if sticker_counts:
            print(f"Average stickers per image: {sum(sticker_counts)/len(sticker_counts):.2f}")


def main():
    parser = argparse.ArgumentParser(description='Synthetic data generator for YOLO training')
    
    parser.add_argument('--stickers_dir', type=str, default='data/raw/stickers',
                        help='Path to directory with stickers')
    parser.add_argument('--backgrounds_dir', type=str, default='data/raw/backgrounds',
                        help='Path to directory with backgrounds')
    parser.add_argument('--output_dir', type=str, default='data/synth/imgs',
                        help='Path to directory for saving generated images')
    parser.add_argument('--annotations_file', type=str, default='annotations.txt',
                        help='File name for saving annotations')
    parser.add_argument('--num_images', type=int, default=1000,
                        help='Number of images to generate')
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
        overlap_ratio=args.overlap_ratio,
        random_seed=args.random_seed
    )
    
    generator.generate_dataset()


if __name__ == "__main__":
    main()
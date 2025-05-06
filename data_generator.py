#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import random
import math
from pathlib import Path
from typing import List, Tuple, Dict
import shutil
from tqdm import tqdm
from image_transformations import ImageTransformer

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
        self.stickers_dir = stickers_dir
        self.backgrounds_dir = backgrounds_dir
        self.output_dir = output_dir
        self.annotations_file = annotations_file
        self.num_images = num_images
        self.overlap_ratio = overlap_ratio
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        os.makedirs(output_dir, exist_ok=True)
        
        self.stickers = self._load_stickers()
        self.backgrounds = self._load_backgrounds()
        print(f"Loaded {len(self.stickers)} stickers and {len(self.backgrounds)} backgrounds")
        
        self.transformer = ImageTransformer()

    def _load_stickers(self) -> List[Dict]:
        stickers = []
        for file in os.listdir(self.stickers_dir):
            if file.lower().endswith(('.png')):
                path = os.path.join(self.stickers_dir, file)
                img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
                if img is None:
                    print(f"Error loading sticker: {path}")
                    continue
                
                if img.shape[2] == 3:
                    alpha = np.ones((img.shape[0], img.shape[1], 1), dtype=img.dtype) * 255
                    img = np.concatenate((img, alpha), axis=2)
                
                class_name = os.path.splitext(file)[0]
                stickers.append({
                    'name': class_name,
                    'image': img,
                    'path': path
                })
        return stickers

    def _load_backgrounds(self) -> List[np.ndarray]:
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

    def _calculate_bbox(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return (0, 0, 0, 0)
        
        x1 = np.min(x_indices)
        y1 = np.min(y_indices)
        x2 = np.max(x_indices)
        y2 = np.max(y_indices)
        
        return (x1, y1, x2, y2)

    def _calculate_overlap(self, bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
        if (bbox1[2] < bbox2[0] or bbox1[0] > bbox2[2] or
            bbox1[3] < bbox2[1] or bbox1[1] > bbox2[3]):
            return 0.0
        
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        min_area = min(bbox1_area, bbox2_area)
        if min_area == 0:
            return 0.0
        
        return intersection_area / min_area

    def _place_sticker(self, background: np.ndarray, sticker: Dict,
                      placed_bboxes: List[Tuple[int, int, int, int]]) -> Tuple[np.ndarray, Tuple[int, int, int, int], str]:
        max_attempts = 30
        for _ in range(max_attempts):
            transformed_sticker, mask = self.transformer.transform_sticker(sticker['image'], background)
            
            if np.sum(mask > 0) < 100:
                continue
            
            bbox_sticker = self._calculate_bbox(mask)
            sticker_width = bbox_sticker[2] - bbox_sticker[0]
            sticker_height = bbox_sticker[3] - bbox_sticker[1]
            
            if sticker_width < 20 or sticker_height < 20:
                continue
            if sticker_width > background.shape[1] * 0.8 or sticker_height > background.shape[0] * 0.8:
                continue
            
            max_x = background.shape[1] - sticker_width
            max_y = background.shape[0] - sticker_height
            if max_x <= 0 or max_y <= 0:
                continue
            x_pos = random.randint(0, max_x)
            y_pos = random.randint(0, max_y)
            
            new_bbox = (
                x_pos + bbox_sticker[0],
                y_pos + bbox_sticker[1],
                x_pos + bbox_sticker[2],
                y_pos + bbox_sticker[3]
            )
            
            overlap_too_much = False
            for placed_bbox in placed_bboxes:
                overlap = self._calculate_overlap(new_bbox, placed_bbox)
                if overlap > self.overlap_ratio:
                    overlap_too_much = True
                    break
            if overlap_too_much:
                continue
            
            roi = background[
                y_pos:y_pos + transformed_sticker.shape[0],
                x_pos:x_pos + transformed_sticker.shape[1]
            ]
            
            if roi.shape[0] < transformed_sticker.shape[0] or roi.shape[1] < transformed_sticker.shape[1]:
                continue
            
            roi_copy = roi.copy()
            
            for i in range(3):
                roi_copy[:, :, i] = (
                    roi[:, :, i] * (1 - transformed_sticker[:, :, 3] / 255.0) +
                    transformed_sticker[:, :, i] * (transformed_sticker[:, :, 3] / 255.0)
                )
            
            background[
                y_pos:y_pos + transformed_sticker.shape[0],
                x_pos:x_pos + transformed_sticker.shape[1]
            ] = roi_copy
            
            return background, new_bbox, sticker['name']
        
        return background, None, None

    def generate_image(self, img_index: int, background_img: np.ndarray) -> str:
        if len(self.stickers) < 3:
            print("Error: Need at least 3 stickers to generate images")
            return ""
            
        background = background_img.copy()
        
        placed_bboxes = []
        annotations = []
        
        placed_sticker_indices = set()
        
        max_placement_attempts = 50
        
        for attempt in range(max_placement_attempts):
            if len(placed_bboxes) == 3:
                break
                
            available_indices = [i for i in range(len(self.stickers)) if i not in placed_sticker_indices]
            
            if not available_indices:
                available_indices = list(range(len(self.stickers)))
                
            sticker_idx = random.choice(available_indices)
            sticker = self.stickers[sticker_idx]
            
            background, bbox, class_name = self._place_sticker(background, sticker, placed_bboxes)
            
            if bbox is not None:
                placed_bboxes.append(bbox)
                placed_sticker_indices.add(sticker_idx)
                
                annotation = f"({bbox[0]},{bbox[1]})-({bbox[2]},{bbox[3]}) {class_name}"
                annotations.append(annotation)
        
        if len(annotations) == 3:
            output_path = os.path.join(self.output_dir, f"{img_index}.jpg")
            cv2.imwrite(output_path, background)
            
            annotation_line = f"{img_index}.jpg " + " ".join(annotations)
            return annotation_line
        else:
            return ""

    def generate_dataset(self):
        annotations_path = os.path.join(self.output_dir, self.annotations_file)
        
        with open(annotations_path, 'w', encoding='utf-8') as f:
            img_idx = 0
            generated_count = 0
            
            pbar = tqdm(total=self.num_images, desc="Generating images")
            bg_idx = 0
            
            while generated_count < self.num_images and bg_idx < len(self.backgrounds):
                background = self.backgrounds[bg_idx]
                
                annotation = self.generate_image(img_idx, background)
                
                if annotation:
                    f.write(annotation + '\n')
                    img_idx += 1
                    generated_count += 1
                    pbar.update(1)
                
                bg_idx += 1
                
                if bg_idx >= len(self.backgrounds) and generated_count < self.num_images:
                    print("Used all backgrounds, restarting from the beginning")
                    bg_idx = 0
                    random.shuffle(self.backgrounds)
            
            pbar.close()
        
        print(f"Dataset generated with {generated_count} images and saved to {self.output_dir}")
        print(f"Annotations saved to {annotations_path}")
        
        self._validate_annotations(annotations_path)
    
    def _validate_annotations(self, annotations_path: str):
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
            
            img_name = parts[0]
            
            annotation_parts = line.strip().split(' ')[1:]
            annotation_count = 0
            
            for part in annotation_parts:
                if ')' in part and '(' in part:
                    annotation_count += 1
            
            sticker_counts.append(annotation_count)
            
            if annotation_count != 3:
                print(f"Warning: Image {img_name} has {annotation_count} stickers, expected 3")
                invalid_count += 1
            else:
                valid_count += 1
        
        print(f"Validation complete: {valid_count} valid images, {invalid_count} invalid images")
        if sticker_counts:
            print(f"Average stickers per image: {sum(sticker_counts)/len(sticker_counts):.2f}")
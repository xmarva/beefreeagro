#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import cv2
import numpy as np
import random
import math
from typing import Tuple

class ImageTransformer:
    def _get_background_color_info(self, background: np.ndarray) -> Tuple[np.ndarray, float]:
        hsv_bg = cv2.cvtColor(background, cv2.COLOR_BGR2HSV)
        
        avg_h = np.mean(hsv_bg[:, :, 0])
        avg_s = np.mean(hsv_bg[:, :, 1])
        avg_v = np.mean(hsv_bg[:, :, 2])
        
        brightness = avg_v / 255.0
        
        return np.array([avg_h, avg_s, avg_v]), brightness

    def _apply_perspective_transform(self, image: np.ndarray) -> np.ndarray:
        height, width = image.shape[:2]
        
        h_skew = np.random.normal(0, 0.4)
        v_skew = np.random.normal(0, 0.4)
        
        h_skew = max(-0.8, min(0.8, h_skew))
        v_skew = max(-0.8, min(0.8, v_skew))
        
        pts1 = np.float32([
            [0, 0],
            [width, 0],
            [0, height],
            [width, height]
        ])
        
        pts2 = np.float32([
            [width * max(0, -h_skew), height * max(0, -v_skew)],
            [width * (1 + max(0, h_skew)), height * max(0, -v_skew)],
            [width * max(0, -h_skew), height * (1 + max(0, v_skew))],
            [width * (1 + max(0, h_skew)), height * (1 + max(0, v_skew))]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        
        transformed = cv2.warpPerspective(
            image,
            M,
            (int(width * 1.5), int(height * 1.5)),
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0, 0)
        )
        
        alpha = transformed[:, :, 3]
        coords = cv2.findNonZero(alpha)
        if coords is None or len(coords) == 0:
            return image
        
        x, y, w, h = cv2.boundingRect(coords)
        cropped = transformed[y:y+h, x:x+w]
        return cropped

    def _add_noise(self, image: np.ndarray, noise_level: float = 0.1) -> np.ndarray:
        result = image.copy()
        
        noise = np.random.normal(0, noise_level * 255, result.shape[:3])
        
        result[:, :, :3] = np.clip(result[:, :, :3] + noise[:, :, :3], 0, 255).astype(np.uint8)
        return result

    def _apply_cutout(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if np.sum(mask) == 0:
            return image, mask
            
        num_cutouts = random.randint(1, 3)
        
        result = image.copy()
        result_mask = mask.copy()
        
        y_indices, x_indices = np.where(mask > 0)
        if len(y_indices) == 0 or len(x_indices) == 0:
            return image, mask
            
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        
        width = max_x - min_x
        height = max_y - min_y
        
        for _ in range(num_cutouts):
            cutout_width = int(random.uniform(0.05, 0.15) * width)
            cutout_height = int(random.uniform(0.05, 0.15) * height)
            
            cutout_width = max(5, cutout_width)
            cutout_height = max(5, cutout_height)
            
            x = random.randint(min_x, max(min_x, max_x - cutout_width))
            y = random.randint(min_y, max(min_y, max_y - cutout_height))
            
            cutout_region = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
            cutout_region[y:y+cutout_height, x:x+cutout_width] = 1
            
            cutout_area = cutout_region & (mask > 0)
            
            result[:, :, 3][cutout_area > 0] = 0
            result_mask[cutout_area > 0] = 0
            
        return result, result_mask

    def _smooth_edges(self, image: np.ndarray) -> np.ndarray:
        result = image.copy()
        
        alpha = result[:, :, 3].copy()
        
        blurred_alpha = cv2.GaussianBlur(alpha, (5, 5), 0)
        
        edges = cv2.Canny(blurred_alpha, 50, 150)
        
        kernel = np.ones((3, 3), np.uint8)
        edge_area = cv2.dilate(edges, kernel, iterations=2)
        
        gradient_mask = cv2.GaussianBlur(alpha, (9, 9), 0)
        
        alpha[edge_area > 0] = gradient_mask[edge_area > 0]
        
        result[:, :, 3] = alpha
        
        return result

    def transform_sticker(self, sticker: np.ndarray, background: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        bg_color_info, bg_brightness = self._get_background_color_info(background)
        
        bg_area = background.shape[0] * background.shape[1]
        target_sticker_area_ratio = np.random.normal(0.03, 0.02)
        
        target_sticker_area_ratio = max(0.0005, min(0.1, target_sticker_area_ratio))
        
        original_sticker_area = sticker.shape[0] * sticker.shape[1]
        target_sticker_area = bg_area * target_sticker_area_ratio
        scale = math.sqrt(target_sticker_area / original_sticker_area)
        
        new_height = int(sticker.shape[0] * scale)
        new_width = int(sticker.shape[1] * scale)
        if new_height < 1 or new_width < 1:
            return np.zeros((1, 1, 4), dtype=np.uint8), np.zeros((1, 1), dtype=np.uint8)
            
        resized = cv2.resize(sticker, (new_width, new_height))
        
        if random.random() < 0.9:
            resized = self._apply_perspective_transform(resized)
            
        if resized.size == 0:
            resized = cv2.resize(sticker, (new_width, new_height))
            
        angle = random.uniform(0, 360)
        center = (resized.shape[1] // 2, resized.shape[0] // 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        abs_cos = abs(rotation_matrix[0, 0])
        abs_sin = abs(rotation_matrix[0, 1])
        bound_w = int(resized.shape[0] * abs_sin + resized.shape[1] * abs_cos)
        bound_h = int(resized.shape[0] * abs_cos + resized.shape[1] * abs_sin)
        
        rotation_matrix[0, 2] += bound_w / 2 - center[0]
        rotation_matrix[1, 2] += bound_h / 2 - center[1]
        
        rotated = cv2.warpAffine(resized, rotation_matrix, (bound_w, bound_h),
                                borderMode=cv2.BORDER_CONSTANT,
                                borderValue=(0, 0, 0, 0))
        
        mask = rotated[:, :, 3] > 128
        
        if random.random() < 0.7:
            rotated, mask = self._apply_cutout(rotated, mask)
        
        rotated = self._smooth_edges(rotated)
        
        hsv = cv2.cvtColor(rotated[:, :, :3], cv2.COLOR_BGR2HSV)
        
        brightness_factor = 0.6 + bg_brightness * 0.8
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * brightness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.8:
            bg_hue = bg_color_info[0]
            hue_shift = (bg_hue - np.mean(hsv[:, :, 0])) * 0.25
            hsv[:, :, 0] = np.clip(hsv[:, :, 0] + hue_shift, 0, 179).astype(np.uint8)
        
        if bg_color_info[1] > 50:
            overlay = np.ones_like(hsv)
            overlay[:, :, 0] = bg_color_info[0]
            overlay[:, :, 1] = min(100, bg_color_info[1] * 0.5)
            overlay[:, :, 2] = 255
            
            overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_HSV2BGR)
            
            sticker_bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
            
            sticker_bgr = cv2.addWeighted(sticker_bgr, 0.7, overlay_bgr, 0.3, 0)
            
            hsv = cv2.cvtColor(sticker_bgr, cv2.COLOR_BGR2HSV)
        
        rotated_color = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        if bg_brightness < 0.7:
            shadow_strength = 0.3 + (0.7 - bg_brightness) * 0.6
            kernel_size = int(max(5, min(15, rotated.shape[0] * 0.08)))
            if kernel_size % 2 == 0:
                kernel_size += 1
            shadow_mask = cv2.GaussianBlur(rotated[:, :, 3].copy(), (kernel_size, kernel_size), 0)
            shadow = np.zeros_like(rotated_color)
            for i in range(3):
                rotated_color[:, :, i] = np.clip(
                    rotated_color[:, :, i] * (1 - shadow_strength * shadow_mask / 255.0), 0, 255
                ).astype(np.uint8)
        
        if random.random() < 0.3:
            if random.random() < 0.5:
                brightness_boost = random.uniform(1.2, 2.0)
                rotated_color = np.clip(rotated_color * brightness_boost, 0, 255).astype(np.uint8)
            else:
                darkness_factor = random.uniform(0.5, 0.8)
                rotated_color = np.clip(rotated_color * darkness_factor, 0, 255).astype(np.uint8)
        
        if random.random() < 0.4:
            blur_size = random.choice([3, 5, 7, 9])
            rotated_color = cv2.GaussianBlur(rotated_color, (blur_size, blur_size), 0)
        
        if random.random() < 0.1:
            noise_level = random.uniform(0.05, 0.2)
            rotated_color = self._add_noise(rotated_color, noise_level)
        
        dark_pixels = np.all(rotated_color < 30, axis=2)
        
        if np.any(dark_pixels):
            gray_value = random.randint(30, 80)
            gray_color = np.array([gray_value, gray_value, gray_value])
            
            for i in range(3):
                rotated_color[:, :, i][dark_pixels] = gray_color[i]
        
        final = np.zeros((rotated.shape[0], rotated.shape[1], 4), dtype=np.uint8)
        final[:, :, :3] = rotated_color
        final[:, :, 3] = rotated[:, :, 3]
        
        return final, mask.astype(np.uint8) * 255
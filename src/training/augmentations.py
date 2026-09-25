"""
Drone-Specific Augmentations for DJI Matrice 4TD
Author: Ali Naderi

Standard augmentations fail for drone perspective. This module adds
top-down and oblique augmentations.
"""

from typing import Dict
import cv2
import numpy as np
import random

class DroneAugmentations:
    """
    Custom augmentations for drone imagery.
    
    Key differences from ground-level:
    - No large rotation (horizon matters)
    - Perspective warp for oblique view
    - Scale variation for altitude changes
    - Small object preservation (hard-hat is tiny)
    """

    @staticmethod
    def perspective_warp(image, bboxes, p=0.3):
        """Simulate oblique drone view"""
        if random.random() > p:
            return image, bboxes
        
        h, w = image.shape[:2]
        # Small perspective distortion
        pts1 = np.float32([[0,0], [w,0], [0,h], [w,h]])
        dw = int(w * 0.05)
        dh = int(h * 0.05)
        pts2 = np.float32([
            [random.randint(-dw, dw), random.randint(-dh, dh)],
            [w - random.randint(-dw, dw), random.randint(-dh, dh)],
            [random.randint(-dw, dw), h - random.randint(-dh, dh)],
            [w - random.randint(-dw, dw), h - random.randint(-dh, dh)]
        ])
        
        M = cv2.getPerspectiveTransform(pts1, pts2)
        warped = cv2.warpPerspective(image, M, (w, h), borderValue=(114,114,114))
        
        # Transform bboxes (simplified - for full pipeline use albumentations)
        # For YOLO training, this is handled by Ultralytics internally via perspective arg
        return warped, bboxes

    @staticmethod
    def get_ultralytics_overrides() -> Dict:
        """
        Returns augmentation overrides for Ultralytics YOLO training
        Optimized for DJI Matrice 4TD
        """
        return {
            'degrees': 0.0,          # No rotation - horizon critical for drone
            'translate': 0.1,
            'scale': 0.5,            # Altitude variation 20m-80m
            'shear': 0.0,
            'perspective': 0.0005,   # Small oblique
            'flipud': 0.0,           # No vertical flip for drone
            'fliplr': 0.5,           # Horizontal flip OK
            'mosaic': 1.0,           # Helps small objects
            'mixup': 0.0,            # Disable - hurts small hard-hat
            'copy_paste': 0.3,       # Useful for small objects
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
        }

    @staticmethod
    def get_hardhat_specific() -> Dict:
        """Extra augmentation for hard-hat small objects"""
        return {
            'copy_paste': 0.5,       # More copy-paste for tiny hard-hats
            'mosaic': 1.0,
            'scale': 0.9,            # Larger scale range for tiny objects
        }

"""
ICDAR 2015 PyTorch Dataset for EAST

Loads images, parses annotations, and generates ground truth maps on-the-fly.
"""

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from .icdar import ICDARAnnotationParser
from .ground_truth import generate_ground_truth_maps

class ICDARDataset(Dataset):
    """
    PyTorch Dataset for ICDAR 2015 text detection (EAST).
    """
    def __init__(self, images_dir, annotations_dir, transform=None, output_stride=4, shrink_ratio=0.4):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_files = sorted([f for f in os.listdir(images_dir) if f.endswith('.jpg') or f.endswith('.png')])
        self.transform = transform
        self.output_stride = output_stride
        self.shrink_ratio = shrink_ratio
        self.parser = ICDARAnnotationParser()
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        # Load annotation
        ann_name = f"gt_{os.path.splitext(img_name)[0]}.txt"
        ann_path = os.path.join(self.annotations_dir, ann_name)
        text_instances = self.parser.parse_annotation_file(ann_path)
        # Generate ground truth maps
        score_map, geometry_map = generate_ground_truth_maps(
            text_instances, h, w, shrink_ratio=self.shrink_ratio, output_stride=self.output_stride
        )
        # Apply transforms (if any)
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        score_map = torch.from_numpy(score_map).unsqueeze(0).float()
        geometry_map = torch.from_numpy(geometry_map).float()
        return {
            'image': image,
            'score_map': score_map,
            'geometry_map': geometry_map,
            'image_name': img_name
        }

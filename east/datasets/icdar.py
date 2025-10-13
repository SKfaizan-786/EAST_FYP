"""
ICDAR 2015 Dataset Parser

Handles loading and parsing of ICDAR 2015 Robust Reading Competition dataset
for scene text detection. Supports both training and test splits with proper
annotation format parsing and validation.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging
from dataclasses import dataclass

# Setup logging
logger = logging.getLogger(__name__)

@dataclass
class TextInstance:
    """Represents a single text instance with quadrilateral coordinates"""
    quad: np.ndarray  # Shape: (4, 2) - four corner points
    text: str         # Transcription text
    difficult: bool   # Whether this is a difficult/ignored instance
    
    def __post_init__(self):
        """Validate and normalize the quadrilateral"""
        if self.quad.shape != (4, 2):
            raise ValueError(f"Quad must have shape (4, 2), got {self.quad.shape}")
        
        # Ensure clockwise order
        self.quad = self._ensure_clockwise_order(self.quad)
    
    def _ensure_clockwise_order(self, quad: np.ndarray) -> np.ndarray:
        """Ensure quadrilateral points are in clockwise order"""
        # Calculate center point
        center = np.mean(quad, axis=0)
        
        # Calculate angles from center to each point
        angles = np.arctan2(quad[:, 1] - center[1], quad[:, 0] - center[0])
        
        # Sort points by angle (clockwise)
        sorted_indices = np.argsort(-angles)  # Negative for clockwise
        
        return quad[sorted_indices]
    
    def get_bbox(self) -> Tuple[int, int, int, int]:
        """Get axis-aligned bounding box (x_min, y_min, x_max, y_max)"""
        x_coords = self.quad[:, 0]
        y_coords = self.quad[:, 1]
        return (
            int(np.min(x_coords)),
            int(np.min(y_coords)), 
            int(np.max(x_coords)),
            int(np.max(y_coords))
        )
    
    def get_area(self) -> float:
        """Calculate polygon area using shoelace formula"""
        x = self.quad[:, 0]
        y = self.quad[:, 1]
        return 0.5 * abs(sum(x[i] * y[(i + 1) % 4] - x[(i + 1) % 4] * y[i] for i in range(4)))
    
    def is_valid(self, min_area: float = 10.0) -> bool:
        """Check if text instance is valid (non-degenerate, sufficient area)"""
        if self.get_area() < min_area:
            return False
        
        # Check for degenerate cases (collinear points)
        for i in range(4):
            p1 = self.quad[i]
            p2 = self.quad[(i + 1) % 4]  
            p3 = self.quad[(i + 2) % 4]
            
            # Calculate cross product to check collinearity
            v1 = p2 - p1
            v2 = p3 - p1
            cross = np.cross(v1, v2)
            
            if abs(cross) < 1e-6:  # Points are nearly collinear
                return False
        
        return True
    
    @property
    def coordinates(self) -> List[float]:
        """Get coordinates as flat list for compatibility"""
        return self.quad.flatten().tolist()
    
    def get_bounding_box(self) -> Tuple[int, int, int, int]:
        """Alias for get_bbox for compatibility"""
        return self.get_bbox()
    
    def is_valid_quadrilateral(self) -> bool:
        """Alias for is_valid for compatibility"""
        return self.is_valid()


class ICDARAnnotationParser:
    """Parser for ICDAR 2015 annotation files"""
    
    def __init__(self, 
                 encoding: str = 'utf-8-sig',
                 ignore_difficult: bool = True,
                 min_area: float = 10.0):
        """
        Initialize ICDAR annotation parser
        
        Args:
            encoding: File encoding for annotation files
            ignore_difficult: Whether to ignore difficult instances marked with ###
            min_area: Minimum area threshold for valid text instances
        """
        self.encoding = encoding
        self.ignore_difficult = ignore_difficult  
        self.min_area = min_area
        
    def parse_annotation_file(self, annotation_path: Union[str, Path]) -> List[TextInstance]:
        """
        Parse a single ICDAR annotation file
        
        Args:
            annotation_path: Path to the annotation file
            
        Returns:
            List of TextInstance objects
        """
        annotation_path = Path(annotation_path)
        
        if not annotation_path.exists():
            logger.warning(f"Annotation file not found: {annotation_path}")
            return []
        
        text_instances = []
        
        try:
            with open(annotation_path, 'r', encoding=self.encoding) as f:
                lines = f.readlines()
                
            for line_idx, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    continue
                    
                try:
                    text_instance = self._parse_line(line)
                    if text_instance and text_instance.is_valid(self.min_area):
                        # Skip difficult instances if configured
                        if self.ignore_difficult and text_instance.difficult:
                            logger.debug(f"Skipping difficult instance in {annotation_path}:{line_idx}")
                            continue
                        text_instances.append(text_instance)
                    
                except Exception as e:
                    logger.warning(f"Failed to parse line {line_idx} in {annotation_path}: {e}")
                    continue
                    
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error reading {annotation_path}: {e}")
            # Try with different encodings
            for alt_encoding in ['utf-8', 'latin1', 'cp1252']:
                try:
                    with open(annotation_path, 'r', encoding=alt_encoding) as f:
                        lines = f.readlines()
                    logger.info(f"Successfully read {annotation_path} with {alt_encoding} encoding")
                    break
                except:
                    continue
            else:
                logger.error(f"Could not read {annotation_path} with any encoding")
                return []
        
        logger.debug(f"Parsed {len(text_instances)} valid instances from {annotation_path}")
        return text_instances
    
    def _parse_line(self, line: str) -> Optional[TextInstance]:
        """Parse a single line of annotation"""
        # ICDAR format: x1,y1,x2,y2,x3,y3,x4,y4,transcription
        parts = line.split(',')
        
        if len(parts) < 8:
            raise ValueError(f"Invalid line format: expected at least 8 coordinates, got {len(parts)}")
        
        try:
            # Parse coordinates
            coords = [float(x) for x in parts[:8]]
            quad = np.array(coords).reshape(4, 2)
            
            # Parse transcription (everything after 8th comma)
            if len(parts) > 8:
                transcription = ','.join(parts[8:]).strip()
            else:
                transcription = ""
            
            # Check if this is a difficult instance
            difficult = transcription == "###" or transcription.lower() == "###"
            
            return TextInstance(
                quad=quad,
                text=transcription, 
                difficult=difficult
            )
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse coordinates: {e}")
    
    def validate_dataset_structure(self, dataset_path: Union[str, Path]) -> Dict[str, any]:
        """
        Validate ICDAR dataset structure and count files
        
        Args:
            dataset_path: Path to ICDAR dataset root
            
        Returns:
            Dictionary with validation results and statistics
        """
        dataset_path = Path(dataset_path)
        
        results = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        # Check directory structure
        required_dirs = [
            'train/images', 'train/annotations',
            'test/images', 'test/annotations'
        ]
        
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                results['valid'] = False
                results['errors'].append(f"Missing directory: {dir_path}")
        
        if not results['valid']:
            return results
        
        # Count files and check correspondence
        train_images_dir = dataset_path / 'train' / 'images'
        train_annotations_dir = dataset_path / 'train' / 'annotations'
        test_images_dir = dataset_path / 'test' / 'images'
        test_annotations_dir = dataset_path / 'test' / 'annotations'
        
        # Get file lists
        train_images = list(train_images_dir.glob('*.jpg')) + list(train_images_dir.glob('*.png'))
        train_annotations = list(train_annotations_dir.glob('*.txt'))
        test_images = list(test_images_dir.glob('*.jpg')) + list(test_images_dir.glob('*.png'))
        test_annotations = list(test_annotations_dir.glob('*.txt'))
        
        results['statistics'] = {
            'train_images': len(train_images),
            'train_annotations': len(train_annotations),
            'test_images': len(test_images),
            'test_annotations': len(test_annotations)
        }
        
        # Check image-annotation correspondence
        train_image_stems = {f.stem for f in train_images}
        train_annotation_stems = {f.stem.replace('gt_', '') for f in train_annotations}
        
        missing_annotations = train_image_stems - train_annotation_stems
        missing_images = train_annotation_stems - train_image_stems
        
        if missing_annotations:
            results['warnings'].append(f"Images without annotations: {len(missing_annotations)}")
        if missing_images:
            results['warnings'].append(f"Annotations without images: {len(missing_images)}")
        
        # Sample a few annotations to check format
        sample_annotations = train_annotations[:5]
        total_instances = 0
        valid_instances = 0
        
        for ann_file in sample_annotations:
            instances = self.parse_annotation_file(ann_file)
            total_instances += len(instances)
            valid_instances += sum(1 for inst in instances if inst.is_valid(self.min_area))
        
        results['statistics']['sample_total_instances'] = total_instances
        results['statistics']['sample_valid_instances'] = valid_instances
        
        if total_instances > 0:
            results['statistics']['valid_instance_ratio'] = valid_instances / total_instances
        
        logger.info(f"Dataset validation complete: {results['statistics']}")
        return results
    
    def validate_dataset_structure(self, images_dir: Union[str, Path], annotations_dir: Union[str, Path]) -> bool:
        """
        Simple validation method for images and annotations directories
        
        Args:
            images_dir: Path to images directory
            annotations_dir: Path to annotations directory
            
        Returns:
            True if validation passes, False otherwise
        """
        images_path = Path(images_dir)
        annotations_path = Path(annotations_dir)
        
        # Check if directories exist
        if not images_path.exists():
            logger.error(f"Images directory not found: {images_path}")
            return False
        
        if not annotations_path.exists():
            logger.error(f"Annotations directory not found: {annotations_path}")
            return False
        
        # Get file lists
        image_files = set(f.stem for f in images_path.glob("*.jpg"))
        image_files.update(f.stem for f in images_path.glob("*.png"))
        
        annotation_files = set()
        for f in annotations_path.glob("gt_*.txt"):
            # Remove 'gt_' prefix to match image names
            name = f.stem[3:] if f.stem.startswith('gt_') else f.stem
            annotation_files.add(name)
        
        # Check correspondence
        missing_annotations = image_files - annotation_files
        missing_images = annotation_files - image_files
        
        if missing_annotations:
            logger.warning(f"Images without annotations: {len(missing_annotations)}")
        
        if missing_images:
            logger.warning(f"Annotations without images: {len(missing_images)}")
        
        # Validation passes if we have both directories and some matching files
        matching_files = len(image_files & annotation_files)
        logger.info(f"Found {matching_files} matching image-annotation pairs")
        
        return matching_files > 0


# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize parser
    parser = ICDARAnnotationParser()
    
    # Test with sample data (if available)
    dataset_path = Path("data/icdar2015")
    
    if dataset_path.exists():
        # Validate dataset structure
        validation_results = parser.validate_dataset_structure(dataset_path)
        
        if validation_results['valid']:
            print("✅ Dataset structure is valid")
            print(f"📊 Statistics: {validation_results['statistics']}")
        else:
            print("❌ Dataset validation failed")
            for error in validation_results['errors']:
                print(f"  - {error}")
        
        if validation_results['warnings']:
            print("⚠️  Warnings:")
            for warning in validation_results['warnings']:
                print(f"  - {warning}")
    else:
        print("📁 Dataset directory not found. Please organize your ICDAR files first.")
        print("Expected structure:")
        print("data/icdar2015/")
        print("├── train/images/")
        print("├── train/annotations/") 
        print("├── test/images/")
        print("└── test/annotations/")
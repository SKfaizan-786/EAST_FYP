"""
Ground Truth Map Generation

This module generates ground truth score and geometry maps for EAST training.
The score map indicates text/non-text regions, and geometry maps encode 
text region orientations and sizes.
"""

import numpy as np
import cv2
from typing import List, Tuple, Union, Optional
import math
from pathlib import Path
import logging

from ..utils.coordinates import CoordinateProcessor, quad_to_rbox
from .icdar import TextInstance

logger = logging.getLogger(__name__)


class GroundTruthGenerator:
    """Generates ground truth maps for EAST training"""
    
    def __init__(self, 
                 shrink_ratio: float = 0.4,
                 min_text_size: int = 8,
                 max_text_size: int = 800):
        """
        Initialize ground truth generator
        
        Args:
            shrink_ratio: Ratio to shrink text regions (0.4 means shrink by 60%)
            min_text_size: Minimum text region size to consider
            max_text_size: Maximum text region size to consider
        """
        self.shrink_ratio = shrink_ratio
        self.min_text_size = min_text_size
        self.max_text_size = max_text_size
        self.coord_processor = CoordinateProcessor()
    
    def generate_score_map(self, 
                          text_instances: List[TextInstance],
                          image_height: int,
                          image_width: int,
                          output_stride: int = 4) -> np.ndarray:
        """
        Generate score map indicating text regions
        
        Args:
            text_instances: List of text instances
            image_height: Original image height
            image_width: Original image width  
            output_stride: Downsampling factor (4 for EAST)
            
        Returns:
            Score map of shape (H/4, W/4) with values in [0, 1]
        """
        # Calculate output dimensions
        output_height = image_height // output_stride
        output_width = image_width // output_stride
        
        # Initialize score map
        score_map = np.zeros((output_height, output_width), dtype=np.float32)
        
        for text_instance in text_instances:
            # Skip difficult instances
            if text_instance.difficult:
                continue
            
            # Get quadrilateral points
            quad = text_instance.quad
            
            # Check text size
            bbox = self.coord_processor.quad_to_bbox(quad)
            text_width = bbox.width
            text_height = bbox.height
            
            if (text_width < self.min_text_size or 
                text_height < self.min_text_size or
                text_width > self.max_text_size or 
                text_height > self.max_text_size):
                continue
            
            # Shrink text region
            shrunken_quad = self._shrink_quad(quad, self.shrink_ratio)
            
            if shrunken_quad is None:
                continue
            
            # Scale coordinates to output resolution
            scaled_quad = shrunken_quad / output_stride
            
            # Create mask for this text instance
            mask = self._create_quad_mask(scaled_quad, output_height, output_width)
            
            # Add to score map
            score_map = np.maximum(score_map, mask)
        
        return score_map
    
    def generate_geometry_map(self,
                             text_instances: List[TextInstance], 
                             image_height: int,
                             image_width: int,
                             output_stride: int = 4) -> np.ndarray:
        """
        Generate geometry map with RBOX encoding
        
        Args:
            text_instances: List of text instances
            image_height: Original image height
            image_width: Original image width
            output_stride: Downsampling factor
            
        Returns:
            Geometry map of shape (H/4, W/4, 5) with [top, right, bottom, left, angle]
        """
        # Calculate output dimensions
        output_height = image_height // output_stride
        output_width = image_width // output_stride
        
        # Initialize geometry map [top, right, bottom, left, angle]
        geometry_map = np.zeros((output_height, output_width, 5), dtype=np.float32)
        
        for text_instance in text_instances:
            # Skip difficult instances
            if text_instance.difficult:
                continue
            
            # Get quadrilateral points
            quad = text_instance.quad
            
            # Check text size
            bbox = self.coord_processor.quad_to_bbox(quad)
            text_width = bbox.width
            text_height = bbox.height
            
            if (text_width < self.min_text_size or 
                text_height < self.min_text_size or
                text_width > self.max_text_size or 
                text_height > self.max_text_size):
                continue
            
            # Shrink text region
            shrunken_quad = self._shrink_quad(quad, self.shrink_ratio)
            
            if shrunken_quad is None:
                continue
            
            # Scale coordinates to output resolution
            scaled_quad = shrunken_quad / output_stride
            original_quad = quad / output_stride
            
            # Generate geometry values for this text region
            self._fill_geometry_map(geometry_map, scaled_quad, original_quad, 
                                  output_height, output_width)
        
        return geometry_map
    
    def _shrink_quad(self, quad: np.ndarray, shrink_ratio: float) -> Optional[np.ndarray]:
        """
        Shrink quadrilateral towards its center
        
        Args:
            quad: Original quadrilateral points (4, 2)
            shrink_ratio: Shrinking ratio
            
        Returns:
            Shrunken quadrilateral or None if invalid
        """
        # Calculate centroid
        center = np.mean(quad, axis=0)
        
        # Shrink each point towards center
        shrunken_quad = center + shrink_ratio * (quad - center)
        
        # Validate shrunken quadrilateral
        if not self.coord_processor.is_valid_quad(shrunken_quad, min_area=4.0):
            return None
        
        return shrunken_quad
    
    def _create_quad_mask(self, 
                         quad: np.ndarray, 
                         height: int, 
                         width: int) -> np.ndarray:
        """
        Create binary mask for quadrilateral region
        
        Args:
            quad: Quadrilateral points (4, 2)
            height: Mask height
            width: Mask width
            
        Returns:
            Binary mask of shape (height, width)
        """
        mask = np.zeros((height, width), dtype=np.float32)
        
        # Convert to integer coordinates
        quad_int = np.round(quad).astype(np.int32)
        
        # Clip to image boundaries
        quad_int[:, 0] = np.clip(quad_int[:, 0], 0, width - 1)
        quad_int[:, 1] = np.clip(quad_int[:, 1], 0, height - 1)
        
        # Fill polygon
        cv2.fillPoly(mask, [quad_int], 1.0)
        
        return mask
    
    def _fill_geometry_map(self,
                          geometry_map: np.ndarray,
                          shrunken_quad: np.ndarray,
                          original_quad: np.ndarray,
                          height: int,
                          width: int):
        """
        Fill geometry map with distance and angle values
        
        Args:
            geometry_map: Geometry map to fill (H, W, 5)
            shrunken_quad: Shrunken quadrilateral for positive region
            original_quad: Original quadrilateral for distance calculation
            height: Map height
            width: Map width
        """
        # Create mask for positive region
        mask = self._create_quad_mask(shrunken_quad, height, width)
        
        # Get positive pixel coordinates
        positive_coords = np.where(mask > 0)
        
        if len(positive_coords[0]) == 0:
            return
        
        # Convert original quad to oriented bounding box
        rbox = quad_to_rbox(original_quad)
        
        # Calculate geometry values for each positive pixel
        for i in range(len(positive_coords[0])):
            y, x = positive_coords[0][i], positive_coords[1][i]
            
            # Calculate distances to quad edges and angle
            distances, angle = self._calculate_pixel_geometry(
                np.array([x, y]), original_quad, rbox
            )
            
            # Store in geometry map [top, right, bottom, left, angle]
            geometry_map[y, x, :4] = distances
            geometry_map[y, x, 4] = angle
    
    def _calculate_pixel_geometry(self,
                                 pixel: np.ndarray,
                                 quad: np.ndarray,
                                 rbox) -> Tuple[np.ndarray, float]:
        """
        Calculate geometry values for a pixel
        
        Args:
            pixel: Pixel coordinates (x, y)
            quad: Original quadrilateral points
            rbox: Oriented bounding box
            
        Returns:
            Tuple of (distances to 4 edges, rotation angle)
        """
        # Calculate distances to each edge of the quadrilateral
        distances = np.zeros(4, dtype=np.float32)
        
        for i in range(4):
            # Get edge points
            p1 = quad[i]
            p2 = quad[(i + 1) % 4]
            
            # Calculate distance from pixel to edge
            distances[i] = self._point_to_line_distance(pixel, p1, p2)
        
        # Use angle from oriented bounding box
        angle = rbox.angle
        
        return distances, angle
    
    def _point_to_line_distance(self, 
                               point: np.ndarray,
                               line_start: np.ndarray,
                               line_end: np.ndarray) -> float:
        """
        Calculate distance from point to line segment
        
        Args:
            point: Point coordinates
            line_start: Line start point
            line_end: Line end point
            
        Returns:
            Distance to line segment
        """
        # Vector from line_start to line_end
        line_vec = line_end - line_start
        line_len = np.linalg.norm(line_vec)
        
        if line_len < 1e-6:
            # Degenerate line, return distance to point
            return np.linalg.norm(point - line_start)
        
        # Unit vector along line
        line_unit = line_vec / line_len
        
        # Vector from line_start to point
        point_vec = point - line_start
        
        # Project point onto line
        projection_len = np.dot(point_vec, line_unit)
        
        # Clamp projection to line segment
        projection_len = np.clip(projection_len, 0, line_len)
        
        # Find closest point on line segment
        closest_point = line_start + projection_len * line_unit
        
        # Return distance
        return np.linalg.norm(point - closest_point)
    
    def visualize_ground_truth(self,
                              image: np.ndarray,
                              score_map: np.ndarray,
                              geometry_map: np.ndarray,
                              output_stride: int = 4) -> np.ndarray:
        """
        Create visualization of ground truth maps
        
        Args:
            image: Original image
            score_map: Score map
            geometry_map: Geometry map
            output_stride: Downsampling factor
            
        Returns:
            Visualization image
        """
        # Resize score map to image size
        score_resized = cv2.resize(score_map, 
                                 (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        
        # Create colored overlay
        score_colored = cv2.applyColorMap(
            (score_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        
        # Blend with original image
        alpha = 0.6
        visualization = cv2.addWeighted(image, 1-alpha, score_colored, alpha, 0)
        
        return visualization


def generate_ground_truth_maps(text_instances: List[TextInstance],
                              image_height: int, 
                              image_width: int,
                              shrink_ratio: float = 0.4,
                              output_stride: int = 4) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to generate both score and geometry maps
    
    Args:
        text_instances: List of text instances
        image_height: Image height
        image_width: Image width
        shrink_ratio: Text region shrinking ratio
        output_stride: Downsampling factor
        
    Returns:
        Tuple of (score_map, geometry_map)
    """
    generator = GroundTruthGenerator(shrink_ratio=shrink_ratio)
    
    score_map = generator.generate_score_map(
        text_instances, image_height, image_width, output_stride
    )
    
    geometry_map = generator.generate_geometry_map(
        text_instances, image_height, image_width, output_stride  
    )
    
    return score_map, geometry_map
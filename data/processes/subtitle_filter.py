"""
Subtitle filtering at data loading stage.
Filters polygons based on geometric cues (aspect ratio, flatness, color consistency)
before creating gt and mask.
"""

import numpy as np
import cv2
from shapely.geometry import Polygon

from concern.config import Configurable, State
from .data_process import DataProcess


class SubtitleFilter(DataProcess):
    """
    Filter polygons based on subtitle characteristics at data loading stage.
    This is more efficient than filtering during training.
    
    Filters polygons based on:
    1. Aspect ratio (horizontal structure)
    2. Flatness (centerline height consistency)
    3. Color consistency (dominant color presence)
    
    Only polygons that satisfy all criteria are kept (marked as not ignored).
    """
    aspect_ratio_threshold = State(default=2.0)
    tilt_threshold = State(default=0.3)  # Coefficient of variation threshold for detecting tilted/rotated text
    wobble_threshold = State(default=5.0)  # Standard deviation threshold for detecting wobbling/wavy text
    color_variance_threshold = State(default=0.1)
    
    def check_aspect_ratio(self, polygon):
        """Check if polygon has horizontal aspect ratio (width >> height)."""
        height = min(np.linalg.norm(polygon[0] - polygon[3]),
                     np.linalg.norm(polygon[1] - polygon[2]))
        width = min(np.linalg.norm(polygon[0] - polygon[1]),
                    np.linalg.norm(polygon[2] - polygon[3]))
        
        if height == 0:
            return False
        
        aspect_ratio = width / height
        return aspect_ratio >= self.aspect_ratio_threshold
    
    def check_flatness(self, polygon):
        """
        Check if polygon has flat centerline (low vertical variance).
        Also checks if y-coordinate counts per x are uniform (horizontal structure).
        
        Criteria:
        1. Y-coordinate counts per x are uniform (not tilted/rotated) - uses tilt_threshold
        2. Centerline Y coordinates have low variance (not wobbling) - uses wobble_threshold
        """
        # Get bounding box
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        x_min, x_max = int(x_coords.min()), int(x_coords.max())
        y_min, y_max = int(y_coords.min()), int(y_coords.max())
        
        # Create a temporary mask to sample centerline
        temp_mask = np.zeros((y_max - y_min + 1, x_max - x_min + 1), dtype=np.uint8)
        polygon_relative = polygon.copy()
        polygon_relative[:, 0] -= x_min
        polygon_relative[:, 1] -= y_min
        cv2.fillPoly(temp_mask, [polygon_relative.astype(np.int32)], 1)
        
        width = x_max - x_min + 1
        if width < self.min_text_size:
            return False
        
        # Compute y-coordinate counts for each x position (vectorized operation)
        y_counts_per_x = temp_mask.sum(axis=0)  # Sum along y-axis for each x
        
        # Filter out x positions with zero y-coordinates
        active_x_indices = np.where(y_counts_per_x > 0)[0]
        y_counts_active = y_counts_per_x[active_x_indices]
        
        if len(y_counts_active) < 2:                                                                # @@ 2
            return False
        
        # Check 1: Y-coordinate count uniformity (each x should have similar number of y pixels)
        # This detects tilted/rotated text where y count varies significantly
        mean_y_count = np.mean(y_counts_active)
        std_y_count = np.std(y_counts_active)
        
        # Coefficient of variation (CV) = std / mean (normalized variance)
        # Low CV means uniform y counts across x (horizontal)
        # High CV means y count varies a lot (tilted/rotated)
        if mean_y_count > 0:
            cv_y_count = std_y_count / mean_y_count
            # If y count varies too much, polygon is likely tilted/rotated
            if cv_y_count > self.tilt_threshold:
                return False
        
        # Check 2: Centerline flatness (low vertical variance)
        centerline_y = []
        for x_pos in active_x_indices:
            y_active = np.where(temp_mask[:, x_pos] > 0)[0]
            if len(y_active) > 0:
                # Use center of vertical span as centerline
                center_y = (y_active.min() + y_active.max()) / 2.0
                centerline_y.append(center_y)
        
        if len(centerline_y) < 2:
            return False
        
        # Compute standard deviation of centerline Y coordinates
        centerline_y = np.array(centerline_y)
        std_centerline = np.std(centerline_y)
        
        return std_centerline < self.wobble_threshold
    
    def check_color_consistency(self, polygon, image):
        """Check if polygon has consistent color (low RGB variance)."""
        # Create mask for this polygon
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [polygon.astype(np.int32)], 1)
        
        # Extract RGB values within polygon
        if image.ndim == 3:
            r_values = image[:, :, 0][mask > 0]
            g_values = image[:, :, 1][mask > 0]
            b_values = image[:, :, 2][mask > 0]
        else:
            # Grayscale
            r_values = g_values = b_values = image[mask > 0]
        
        if len(r_values) < 10:  # Too few pixels
            return False
        
        # Compute variance
        r_var = np.var(r_values)
        g_var = np.var(g_values)
        b_var = np.var(b_values)
        avg_variance = (r_var + g_var + b_var) / 3.0
        
        return avg_variance < self.color_variance_threshold
    
    def filter_polygons(self, polygons, ignore_tags, image):
        """
        Filter polygons based on subtitle characteristics.
        
        Args:
            polygons: numpy array of shape (N, 4, 2) or list of arrays
            ignore_tags: numpy array of shape (N,)
            image: numpy array of shape (H, W, C)
            
        Returns:
            filtered_polygons: filtered polygons
            filtered_ignore_tags: filtered ignore tags (True for non-subtitle regions)
        """
        if len(polygons) == 0:
            return polygons, ignore_tags
        
        # Convert to list if needed
        if isinstance(polygons, np.ndarray):
            polygon_list = [polygons[i] for i in range(len(polygons))]
        else:
            polygon_list = list(polygons)
        
        filtered_polygons = []
        filtered_ignore_tags = []
        
        for i, polygon in enumerate(polygon_list):
            # Skip already ignored polygons
            if ignore_tags[i]:
                filtered_polygons.append(polygon)
                filtered_ignore_tags.append(True)
                continue
            
            # Check each criterion
            has_aspect_ratio = self.check_aspect_ratio(polygon)
            has_flatness = self.check_flatness(polygon)
            has_color_consistency = self.check_color_consistency(polygon, image)
            
            # Keep only if all criteria are satisfied (subtitle-like)
            is_subtitle = has_aspect_ratio and has_flatness and has_color_consistency
            
            filtered_polygons.append(polygon)
            filtered_ignore_tags.append(not is_subtitle)  # True if not subtitle
        
        # Convert back to numpy array if original was numpy
        if isinstance(polygons, np.ndarray):
            filtered_polygons = np.array(filtered_polygons)
            filtered_ignore_tags = np.array(filtered_ignore_tags, dtype=np.uint8)
        else:
            filtered_ignore_tags = np.array(filtered_ignore_tags, dtype=np.uint8)
        
        return filtered_polygons, filtered_ignore_tags
    
    def process(self, data):
        """
        Filter polygons in data based on subtitle characteristics.
        
        Required keys:
            image, polygons, ignore_tags
        Modified keys:
            ignore_tags (updated with filtered results)
        """
        image = data['image']
        polygons = data['polygons']
        ignore_tags = data['ignore_tags']
        
        # Filter polygons
        filtered_polygons, filtered_ignore_tags = self.filter_polygons(
            polygons, ignore_tags, image)
        
        # Update data
        data['polygons'] = filtered_polygons
        data['ignore_tags'] = filtered_ignore_tags
        
        return data


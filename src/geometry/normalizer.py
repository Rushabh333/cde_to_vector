"""Spatial normalization for geometric data."""
import numpy as np
from typing import List, Tuple, Optional
import logging


class Normalizer:
    """Normalize geometric coordinates to unit square [0, 1]²."""
    
    def __init__(
        self,
        target_range: Tuple[float, float] = (0.0, 1.0),
        preserve_aspect_ratio: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the normalizer.
        
        Args:
            target_range: Target coordinate range (min, max)
            preserve_aspect_ratio: Whether to preserve aspect ratio during scaling
            logger: Logger instance
        """
        self.target_range = target_range
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.logger = logger or logging.getLogger(__name__)
    
    def normalize(self, points: np.ndarray) -> np.ndarray:
        """
        Normalize coordinates to the unit square.
        
        Uses the formula:
        V_norm = (V - μ) / (2s) + 0.5
        
        Where:
        - μ is the centroid
        - s is the max dimension for scale invariance
        
        Args:
            points: Array of shape (N, 8) containing [sx, sy, cx1, cy1, cx2, cy2, ex, ey]
        
        Returns:
            Normalized points in range [0, 1]
        """
        if points.size == 0:
            return points
        
        # Reshape to (N*4, 2) to process all (x, y) pairs
        n_curves = points.shape[0]
        xy_points = points.reshape(-1, 2)  # Shape: (N*4, 2)
        
        # Compute centroid μ
        centroid = np.mean(xy_points, axis=0)
        
        # Center the points
        centered = xy_points - centroid
        
        # Find max dimension s for scaling
        if self.preserve_aspect_ratio:
            # Use the same scale for both dimensions
            max_dim = np.max(np.abs(centered))
            scale = max_dim if max_dim > 0 else 1.0
        else:
            # Scale each dimension independently
            scale = np.max(np.abs(centered), axis=0)
            scale = np.where(scale > 0, scale, 1.0)
        
        # Normalize: V_norm = (V - μ) / (2s) + 0.5
        if self.preserve_aspect_ratio:
            normalized = centered / (2 * scale) + 0.5
        else:
            normalized = centered / (2 * scale[np.newaxis, :]) + 0.5
        
        # Clip to target range to handle numerical errors
        min_val, max_val = self.target_range
        normalized = np.clip(normalized, min_val, max_val)
        
        # Reshape back to (N, 8)
        normalized_points = normalized.reshape(n_curves, 8)
        
        scale_str = f"{scale:.2f}" if isinstance(scale, (float, np.floating)) else f"{scale[0]:.2f}"
        self.logger.debug(
            f"Normalized {n_curves} curves. "
            f"Centroid: ({centroid[0]:.2f}, {centroid[1]:.2f}), "
            f"Scale: {scale_str}"
        )
        
        return normalized_points.astype(np.float32)

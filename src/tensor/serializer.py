"""Tensor serialization for model-ready output."""
import numpy as np
import torch
from pathlib import Path
from typing import List, Optional
import logging


class TensorSerializer:
    """Serialize Bézier curves to fixed-dimensional tensors."""
    
    def __init__(
        self,
        max_sequence_length: int = 128,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the tensor serializer.
        
        Args:
            max_sequence_length: Maximum number of curves in output tensor
            logger: Logger instance
        """
        self.max_sequence_length = max_sequence_length
        self.logger = logger or logging.getLogger(__name__)
    
    def serialize(
        self,
        control_points_list: List[np.ndarray],
        output_path: Path,
        format: str = "pt"
    ) -> Path:
        """
        Serialize control points to a tensor file.
        
        Output tensor shape: (max_sequence_length, 14)
        
        Feature vector per row:
        [sx, sy, cx1, cy1, cx2, cy2, ex, ey, pen_state, is_line, is_curve, is_move, pad1, pad2]
        
        Args:
            control_points_list: List of arrays, each shape (8,) with control points
            output_path: Path for output file
            format: Output format ('pt' for PyTorch, 'npy' for NumPy)
        
        Returns:
            Path to saved tensor file
        
        Raises:
            ValueError: If format is unsupported
        """
        n_curves = len(control_points_list)
        
        if n_curves == 0:
            self.logger.warning("No curves to serialize, creating empty tensor")
        
        # Initialize tensor with zeros
        tensor = np.zeros((self.max_sequence_length, 14), dtype=np.float32)
        
        # Truncate or pad
        actual_curves = min(n_curves, self.max_sequence_length)
        
        if n_curves > self.max_sequence_length:
            self.logger.warning(
                f"Truncating {n_curves} curves to {self.max_sequence_length}"
            )
        
        # Fill tensor with curve data
        for i in range(actual_curves):
            cp = control_points_list[i]  # Shape: (8,)
            
            # First 8 values: control points [sx, sy, cx1, cy1, cx2, cy2, ex, ey]
            tensor[i, 0:8] = cp
            
            # pen_state: 1 = Draw
            tensor[i, 8] = 1.0
            
            # One-hot encoding of command type
            # For now, all curves are treated as cubic Bézier curves
            tensor[i, 9] = 0.0   # is_line (could be enhanced later)
            tensor[i, 10] = 1.0  # is_curve
            tensor[i, 11] = 0.0  # is_move
            
            # Padding features (reserved for future use)
            tensor[i, 12] = 0.0
            tensor[i, 13] = 0.0
        
        # Save to file
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        if format == "pt":
            # Save as PyTorch tensor
            torch_tensor = torch.from_numpy(tensor)
            torch.save(torch_tensor, output_path)
            self.logger.info(f"Saved PyTorch tensor to {output_path}")
        elif format == "npy":
            # Save as NumPy array
            np.save(output_path, tensor)
            self.logger.info(f"Saved NumPy array to {output_path}")
        else:
            raise ValueError(f"Unsupported format: {format}. Use 'pt' or 'npy'")
        
        self.logger.info(
            f"Serialized {actual_curves}/{n_curves} curves "
            f"to shape {tensor.shape}"
        )
        
        return output_path
    
    def load(self, tensor_path: Path, format: str = "pt") -> np.ndarray:
        """
        Load a tensor from file.
        
        Args:
            tensor_path: Path to tensor file
            format: File format ('pt' or 'npy')
        
        Returns:
            Loaded tensor as NumPy array
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If format is unsupported
        """
        if not tensor_path.exists():
            raise FileNotFoundError(f"Tensor file not found: {tensor_path}")
        
        if format == "pt":
            tensor = torch.load(tensor_path)
            return tensor.numpy()
        elif format == "npy":
            return np.load(tensor_path)
        else:
            raise ValueError(f"Unsupported format: {format}")

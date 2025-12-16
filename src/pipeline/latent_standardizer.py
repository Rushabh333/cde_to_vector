"""Utility for standardizing different latent formats to common dimensions."""
import torch
import numpy as np
from typing import Union, Literal


class LatentStandardizer:
    """Standardize latent representations from different modalities."""
    
    def __init__(self, target_dim: int = 512):
        """
        Initialize the latent standardizer.
        
        Args:
            target_dim: Target dimension for standardized latents
        """
        self.target_dim = target_dim
    
    def standardize_vector(
        self,
        tensor: torch.Tensor,
        method: Literal['flatten', 'pool', 'pad'] = 'flatten'
    ) -> torch.Tensor:
        """
        Standardize vector latent (N, 14) to (target_dim,).
        
        Args:
            tensor: Input tensor of shape (N, 14)
            method: Standardization method
                - 'flatten': Flatten and pad/truncate
                - 'pool': Mean pool then project
                - 'pad': Pad with zeros to match dimension
                
        Returns:
            Standardized tensor of shape (target_dim,)
        """
        if method == 'flatten':
            # Flatten to 1D
            flat = tensor.flatten()
            
            if len(flat) < self.target_dim:
                # Pad with zeros
                padded = torch.zeros(self.target_dim)
                padded[:len(flat)] = flat
                return padded
            else:
                # Truncate
                return flat[:self.target_dim]
        
        elif method == 'pool':
            # Mean pool across curves
            pooled = tensor.mean(dim=0)  # (14,)
            
            # Repeat to match target dimension
            repeats = self.target_dim // 14
            remainder = self.target_dim % 14
            
            result = pooled.repeat(repeats)
            if remainder > 0:
                result = torch.cat([result, pooled[:remainder]])
            
            return result
        
        elif method == 'pad':
            # Pad to target dimension
            N, D = tensor.shape
            total = N * D
            
            flat = tensor.flatten()
            if total < self.target_dim:
                padded = torch.zeros(self.target_dim)
                padded[:total] = flat
                return padded
            else:
                return flat[:self.target_dim]
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def standardize_raster(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Standardize raster latent (1, 512) or (512,) to (target_dim,).
        
        Args:
            tensor: Input tensor of shape (1, 512) or (512,)
            
        Returns:
            Standardized tensor of shape (target_dim,)
        """
        # Flatten if batched
        if tensor.ndim > 1:
            tensor = tensor.flatten()
        
        current_dim = tensor.shape[0]
        
        if current_dim == self.target_dim:
            # Already correct dimension
            return tensor
        
        elif current_dim < self.target_dim:
            # Pad with zeros
            padded = torch.zeros(self.target_dim)
            padded[:current_dim] = tensor
            return padded
        
        else:
            # Truncate
            return tensor[:self.target_dim]
    
    def standardize(
        self,
        tensor: torch.Tensor,
        modality: Literal['vector', 'raster']
    ) -> torch.Tensor:
        """
        Standardize latent from any modality to target dimension.
        
        Args:
            tensor: Input tensor
            modality: 'vector' or 'raster'
            
        Returns:
            Standardized tensor of shape (target_dim,)
        """
        if modality == 'vector':
            return self.standardize_vector(tensor)
        elif modality == 'raster':
            return self.standardize_raster(tensor)
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def normalize(self, tensor: torch.Tensor) -> torch.Tensor:
        """
        Normalize tensor to have unit norm.
        
        Args:
            tensor: Input tensor
            
        Returns:
            Normalized tensor
        """
        norm = torch.norm(tensor)
        if norm > 0:
            return tensor / norm
        else:
            return tensor

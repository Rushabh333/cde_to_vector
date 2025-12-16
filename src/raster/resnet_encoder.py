"""ResNet-based encoder for raster images."""
import torch
import torch.nn as nn
from torchvision import models
from pathlib import Path
from typing import Optional
import logging


class ResNetEncoder(nn.Module):
    """Encode raster images to latent vectors using pretrained ResNet."""
    
    def __init__(
        self,
        latent_dim: int = 512,
        pretrained: bool = True,
        device: str = 'cpu',
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the ResNet encoder.
        
        Args:
            latent_dim: Dimension of output latent vector
            pretrained: Whether to use pretrained ImageNet weights
            device: Device to run on ('cpu' or 'cuda')
            logger: Logger instance
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.device = device
        self.logger = logger or logging.getLogger(__name__)
        
        self.logger.info(f"Initializing ResNet-50 encoder (pretrained={pretrained})")
        
        # Load pretrained ResNet-50
        resnet = models.resnet50(pretrained=pretrained)
        
        # Remove the final classification layer
        # ResNet-50 outputs 2048-dimensional features
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        
        # Add projection layer to desired latent dimension
        self.projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(1024, latent_dim)
        )
        
        # Move to device
        self.to(device)
        
        # Set to evaluation mode by default
        self.eval()
        
        self.logger.info(f"Encoder initialized on {device}, output dim: {latent_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors.
        
        Args:
            x: Input tensor of shape (B, 3, H, W) or (3, H, W)
            
        Returns:
            Latent tensor of shape (B, latent_dim) or (latent_dim,)
        """
        # Handle single image (add batch dimension)
        single_image = False
        if x.ndim == 3:
            x = x.unsqueeze(0)
            single_image = True
        
        # Extract features
        with torch.no_grad():
            features = self.features(x)  # (B, 2048, 1, 1)
        
        # Flatten
        features = features.flatten(1)  # (B, 2048)
        
        # Project to latent dimension
        latent = self.projection(features)  # (B, latent_dim)
        
        # Remove batch dimension if input was single image
        if single_image:
            latent = latent.squeeze(0)
        
        return latent
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode images to latent vectors (alias for forward).
        
        Args:
            x: Input tensor of shape (B, 3, H, W) or (3, H, W)
            
        Returns:
            Latent tensor of shape (B, latent_dim) or (latent_dim,)
        """
        return self.forward(x)
    
    def save_weights(self, path: Path):
        """
        Save encoder weights.
        
        Args:
            path: Path to save weights
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'projection_state_dict': self.projection.state_dict(),
            'latent_dim': self.latent_dim
        }, path)
        self.logger.info(f"Saved encoder weights to {path}")
    
    def load_weights(self, path: Path):
        """
        Load encoder weights.
        
        Args:
            path: Path to saved weights
        """
        if not path.exists():
            raise FileNotFoundError(f"Weights file not found: {path}")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.projection.load_state_dict(checkpoint['projection_state_dict'])
        self.logger.info(f"Loaded encoder weights from {path}")

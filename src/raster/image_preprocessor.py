"""Image preprocessing for raster inputs."""
from pathlib import Path
from typing import Union, Tuple
import torch
from torchvision import transforms
from PIL import Image
import logging

# Import validators
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from utils.validators import ImageValidator, ValidationError


class ImagePreprocessor:
    """Preprocess raster images for neural network encoding."""
    
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        normalize: bool = True,
        logger: logging.Logger = None
    ):
        """
        Initialize the image preprocessor.
        
        Args:
            target_size: Target image size (height, width)
            normalize: Whether to normalize using ImageNet statistics
            logger: Logger instance
        """
        self.target_size = target_size
        self.logger = logger or logging.getLogger(__name__)
        
        # Build transformation pipeline
        transform_list = [
            transforms.Resize(target_size),
            transforms.CenterCrop(target_size),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ]
        
        if normalize:
            # ImageNet normalization statistics
            transform_list.append(
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            )
        
        self.transform = transforms.Compose(transform_list)
    
    def load_image(self, image_path: Path) -> Image.Image:
        """
        Load image from file and convert to RGB.
        
        Args:
            image_path: Path to image file
            
        Returns:
            PIL Image in RGB format
            
        Raises:
            ValidationError: If image file is invalid or cannot be loaded
        """
        # Validate image first
        try:
            width, height, mode = ImageValidator.validate_image(image_path, max_size_mb=50)
            self.logger.debug(f"Loading image: {width}x{height}, mode={mode}")
        except ValidationError:
            raise  # Re-raise validation errors with helpful messages
        
        try:
            img = Image.open(image_path)
            
            # Convert to RGB (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                self.logger.debug(f"Converting {img.mode} to RGB")
                img = img.convert('RGB')
            
            return img
            
        except ValidationError:
            raise  # Already handled
        except Exception as e:
            # Wrap unexpected errors
            raise ValidationError(
                f"Unexpected error loading image {image_path.name}: {e}",
                suggestion="Check if file is corrupted or in an unusual format",
                error_code="IMAGE_LOAD_ERROR"
            )
    
    def preprocess(self, image_path: Path) -> torch.Tensor:
        """
        Load and preprocess an image.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Preprocessed image tensor of shape (3, H, W)
        """
        self.logger.debug(f"Preprocessing image: {image_path.name}")
        
        # Load image
        img = self.load_image(image_path)
        
        # Apply transformations
        img_tensor = self.transform(img)
        
        self.logger.debug(
            f"Preprocessed to shape {img_tensor.shape}, "
            f"range [{img_tensor.min():.3f}, {img_tensor.max():.3f}]"
        )
        
        return img_tensor
    
    def preprocess_batch(self, image_paths: list[Path]) -> torch.Tensor:
        """
        Preprocess multiple images into a batch.
        
        Args:
            image_paths: List of paths to image files
            
        Returns:
            Batch tensor of shape (B, 3, H, W)
        """
        tensors = [self.preprocess(path) for path in image_paths]
        return torch.stack(tensors)

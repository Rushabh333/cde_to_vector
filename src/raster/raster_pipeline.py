"""Raster image processing pipeline."""
from pathlib import Path
from typing import Optional
import torch
import logging

from .image_preprocessor import ImagePreprocessor
from .resnet_encoder import ResNetEncoder


class RasterPipeline:
    """Pipeline for processing raster images to latent tensors."""
    
    def __init__(
        self,
        latent_dim: int = 512,
        image_size: tuple[int, int] = (256, 256),
        device: str = 'cpu',
        pretrained: bool = True,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the raster processing pipeline.
        
        Args:
            latent_dim: Dimension of output latent vector
            image_size: Target image size (height, width)
            device: Device to run on ('cpu' or 'cuda')
            pretrained: Whether to use pretrained ResNet weights
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        self.latent_dim = latent_dim
        self.device = device
        
        self.logger.info("Initializing Raster Pipeline")
        self.logger.info(f"  Latent dim: {latent_dim}")
        self.logger.info(f"  Image size: {image_size}")
        self.logger.info(f"  Device: {device}")
        
        # Initialize components
        self.preprocessor = ImagePreprocessor(
            target_size=image_size,
            normalize=True,
            logger=self.logger
        )
        
        self.encoder = ResNetEncoder(
            latent_dim=latent_dim,
            pretrained=pretrained,
            device=device,
            logger=self.logger
        )
    
    def process_file(self, image_path: Path, output_path: Path) -> Path:
        """
        Process a single image file through the pipeline.
        
        Args:
            image_path: Path to input image (.png, .jpg, etc.)
            output_path: Path for output tensor file (.pt)
            
        Returns:
            Path to output tensor file
            
        Raises:
            FileNotFoundError: If image file doesn't exist
            ValueError: If image processing fails
        """
        self.logger.info(f"Processing raster image: {image_path.name}")
        
        # Step 1: Preprocess image
        try:
            img_tensor = self.preprocessor.preprocess(image_path)
        except Exception as e:
            self.logger.error(f"Preprocessing failed: {e}")
            raise
        
        # Move to device
        img_tensor = img_tensor.to(self.device)
        
        # Step 2: Encode to latent
        try:
            latent = self.encoder.encode(img_tensor)
        except Exception as e:
            self.logger.error(f"Encoding failed: {e}")
            raise
        
        # Step 3: Save tensor
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Add batch dimension if single image: (512,) -> (1, 512)
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        
        # Move to CPU for saving
        latent_cpu = latent.cpu()
        
        try:
            torch.save(latent_cpu, output_path)
            self.logger.info(
                f"Saved latent tensor to {output_path} "
                f"(shape: {latent_cpu.shape})"
            )
        except Exception as e:
            self.logger.error(f"Failed to save tensor: {e}")
            raise
        
        return output_path
    
    def process_batch(
        self,
        image_paths: list[Path],
        output_dir: Path
    ) -> list[Path]:
        """
        Process multiple images in batch.
        
        Args:
            image_paths: List of paths to image files
            output_dir: Directory for output tensors
            
        Returns:
            List of paths to output tensor files
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths = []
        
        self.logger.info(f"Processing {len(image_paths)} images in batch mode")
        
        for img_path in image_paths:
            output_path = output_dir / f"{img_path.stem}.pt"
            try:
                self.process_file(img_path, output_path)
                output_paths.append(output_path)
            except Exception as e:
                self.logger.error(
                    f"Failed to process {img_path.name}: {e}"
                )
                continue
        
        self.logger.info(
            f"Batch processing complete: "
            f"{len(output_paths)}/{len(image_paths)} successful"
        )
        
        return output_paths

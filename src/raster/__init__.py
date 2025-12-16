"""Raster image processing module for dual-stream pipeline.

This module handles raster image inputs (PNG, JPG, etc.) and encodes them
to latent representations compatible with the vector pipeline.
"""

from .image_preprocessor import ImagePreprocessor
from .resnet_encoder import ResNetEncoder
from .raster_pipeline import RasterPipeline

__all__ = ['ImagePreprocessor', 'ResNetEncoder', 'RasterPipeline']

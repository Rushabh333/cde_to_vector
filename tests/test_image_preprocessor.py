"""Tests for image preprocessor."""
import pytest
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from raster.image_preprocessor import ImagePreprocessor
from utils.validators import ValidationError


class TestImagePreprocessor:
    """Test image preprocessing functionality."""
    
    @pytest.fixture
    def preprocessor(self):
        """Create preprocessor instance."""
        return ImagePreprocessor(target_size=(256, 256), normalize=True)
    
    def test_load_valid_image(self, preprocessor, sample_image):
        """Test loading a valid image."""
        img = preprocessor.load_image(sample_image)
        assert img.mode == 'RGB'
        assert img.size == (256, 256)
    
    def test_load_grayscale_converts_to_rgb(self, preprocessor, sample_grayscale_image):
        """Test that grayscale images are converted to RGB."""
        img = preprocessor.load_image(sample_grayscale_image)
        assert img.mode == 'RGB'
    
    def test_load_corrupted_image_raises_error(self, preprocessor, corrupted_image):
        """Test that corrupted images raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            preprocessor.load_image(corrupted_image)
        
        assert "IMAGE_CORRUPT" in str(exc_info.value)
    
    def test_preprocess_returns_tensor(self, preprocessor, sample_image):
        """Test preprocessing returns tensor with correct shape."""
        tensor = preprocessor.preprocess(sample_image)
        
        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 256, 256)
        assert tensor.dtype == torch.float32
    
    def test_preprocess_normalizes(self, preprocessor, sample_image):
        """Test that preprocessing normalizes values."""
        tensor = preprocessor.preprocess(sample_image)
        
        # With ImageNet normalization, values should be roughly in [-2, 2] range
        assert tensor.min() >= -3
        assert tensor.max() <= 3
    
    def test_preprocess_batch(self, preprocessor, sample_image, sample_small_image):
        """Test batch preprocessing."""
        batch = preprocessor.preprocess_batch([sample_image, sample_small_image])
        
        assert isinstance(batch, torch.Tensor)
        assert batch.shape == (2, 3, 256, 256)
    
    def test_preprocessor_without_normalization(self, sample_image):
        """Test preprocessor without normalization."""
        preprocessor = ImagePreprocessor(target_size=(128, 128), normalize=False)
        tensor = preprocessor.preprocess(sample_image)
        
        assert tensor.shape == (3, 128, 128)
        # Without normalization, values should be in [0, 1]
        assert tensor.min() >= 0
        assert tensor.max() <= 1

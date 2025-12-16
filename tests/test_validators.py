"""Tests for validators module."""
import pytest
from pathlib import Path
from PIL import Image
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.validators import (
    ValidationError,
    ConfigValidator,
    FileValidator,
    ImageValidator,
    validate_input
)


class TestConfigValidator:
    """Test configuration validation."""
    
    def test_valid_config(self, sample_config):
        """Test validation of valid config file."""
        config = ConfigValidator.validate_config(sample_config)
        assert isinstance(config, dict)
        assert 'processing' in config
        assert 'paths' in config
        assert 'logging' in config
    
    def test_missing_config_file(self, test_data_dir):
        """Test error when config file doesn't exist."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_config(test_data_dir / 'nonexistent.yaml')
        
        assert "CONFIG_NOT_FOUND" in str(exc_info.value)
        assert "Suggestion" in str(exc_info.value)
    
    def test_invalid_yaml(self, invalid_config):
        """Test error with invalid YAML syntax."""
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_config(invalid_config)
        
        assert "CONFIG_INVALID_YAML" in str(exc_info.value)
    
    def test_missing_required_section(self, test_data_dir):
        """Test error when required section is missing."""
        import yaml
        
        config_path = test_data_dir / 'incomplete.yaml'
        with open(config_path, 'w') as f:
            yaml.dump({'processing': {'max_sequence_length': 128}}, f)
        
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_config(config_path)
        
        assert "CONFIG_MISSING_SECTION" in str(exc_info.value)
        assert "paths" in str(exc_info.value).lower()
    
    def test_invalid_sequence_length(self, test_data_dir):
        """Test error with invalid max_sequence_length."""
        import yaml
        
        config = {
            'processing': {'max_sequence_length': -1},
            'paths': {'raw_data': '.', 'interim_data': '.', 'processed_data': '.'},
            'logging': {'level': 'INFO', 'log_file': 'test.log'}
        }
        
        config_path = test_data_dir / 'bad_seq.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_config(config_path)
        
        assert "CONFIG_INVALID_VALUE" in str(exc_info.value)


class TestFileValidator:
    """Test file validation."""
    
    def test_validate_existing_file(self, sample_image):
        """Test validation of existing file."""
        FileValidator.validate_file_exists(sample_image)  # Should not raise
    
    def test_validate_nonexistent_file(self, test_data_dir):
        """Test error for nonexistent file."""
        with pytest.raises(ValidationError) as exc_info:
            FileValidator.validate_file_exists(test_data_dir / 'missing.png')
        
        assert "FILE_NOT_FOUND" in str(exc_info.value)
    
    def test_validate_directory_as_file(self, test_data_dir):
        """Test error when directory provided instead of file."""
        with pytest.raises(ValidationError) as exc_info:
            FileValidator.validate_file_exists(test_data_dir)
        
        assert "NOT_A_FILE" in str(exc_info.value)
    
    def test_validate_file_format(self, sample_image):
        """Test file format validation."""
        ext = FileValidator.validate_file_format(
            sample_image,
            allowed_formats={'.png', '.jpg'}
        )
        assert ext == '.png'
    
    def test_invalid_file_format(self, test_data_dir):
        """Test error with unsupported format."""
        fake_file = test_data_dir / 'test.xyz'
        fake_file.touch()
        
        with pytest.raises(ValidationError) as exc_info:
            FileValidator.validate_file_format(
                fake_file,
                allowed_formats={'.png', '.jpg'}
            )
        
        assert "UNSUPPORTED_FORMAT" in str(exc_info.value)
    
    def test_validate_directory(self, test_data_dir):
        """Test directory validation."""
        FileValidator.validate_directory(test_data_dir)  # Should not raise
    
    def test_create_directory(self, test_data_dir):
        """Test directory creation."""
        new_dir = test_data_dir / 'new_directory'
        FileValidator.validate_directory(new_dir, create=True)
        assert new_dir.exists()
        assert new_dir.is_dir()


class TestImageValidator:
    """Test image validation."""
    
    def test_valid_image(self, sample_image):
        """Test validation of valid image."""
        width, height, mode = ImageValidator.validate_image(sample_image)
        assert width == 256
        assert height == 256
        assert mode == 'RGB'
    
    def test_grayscale_image(self, sample_grayscale_image):
        """Test validation of grayscale image."""
        width, height, mode = ImageValidator.validate_image(sample_grayscale_image)
        assert width == 256
        assert height == 256
        assert mode == 'L'
    
    def test_corrupted_image(self, corrupted_image):
        """Test error with corrupted image."""
        with pytest.raises(ValidationError) as exc_info:
            ImageValidator.validate_image(corrupted_image)
        
        assert "IMAGE_CORRUPT" in str(exc_info.value)
    
    def test_image_too_small(self, test_data_dir):
        """Test error with image smaller than minimum size."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        tiny_img = Image.new('RGB', (5, 5))
        tiny_path = img_dir / 'tiny.png'
        tiny_img.save(tiny_path)
        
        with pytest.raises(ValidationError) as exc_info:
            ImageValidator.validate_image(tiny_path)
        
        assert "IMAGE_TOO_SMALL" in str(exc_info.value)


class TestValidateInput:
    """Test the convenience validate_input function."""
    
    def test_detect_raster(self, sample_image):
        """Test auto-detection of raster image."""
        modality = validate_input(sample_image)
        assert modality == 'raster'
    
    def test_unknown_format(self, test_data_dir):
        """Test error with unknown format."""
        unknown_file = test_data_dir / 'test.xyz'
        unknown_file.touch()
        
        with pytest.raises(ValidationError) as exc_info:
            validate_input(unknown_file)
        
        assert "UNKNOWN_FORMAT" in str(exc_info.value)

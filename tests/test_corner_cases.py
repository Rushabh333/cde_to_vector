"""Corner case and edge case tests."""
import pytest
from pathlib import Path
import sys
from PIL import Image
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from utils.validators import ValidationError, FileValidator, validate_input
from raster.image_preprocessor import ImagePreprocessor


class TestFileNameCornerCases:
    """Test handling of unusual filenames."""
    
    def test_unicode_filename(self, test_data_dir):
        """Test file with unicode characters."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        # Create image with unicode name
        img = Image.new('RGB', (256, 256), color='red')
        unicode_path = img_dir / '测试_图片_émoji🎨.png'
        img.save(unicode_path)
        
        # Should detect as raster
        modality = validate_input(unicode_path)
        assert modality == 'raster'
    
    def test_spaces_in_filename(self, test_data_dir):
        """Test file with spaces in name."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        img = Image.new('RGB', (256, 256))
        spaced_path = img_dir / 'my test image with spaces.png'
        img.save(spaced_path)
        
        preprocessor = ImagePreprocessor()
        tensor = preprocessor.preprocess(spaced_path)
        assert tensor.shape == (3, 256, 256)
    
    def test_mixed_case_extension(self, test_data_dir):
        """Test file with mixed case extension."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        img = Image.new('RGB', (100, 100))
        mixed_path = img_dir / 'test.PNG'  # Uppercase extension
        img.save(mixed_path)
        
        # Should still detect correctly
        modality = validate_input(mixed_path)
        assert modality == 'raster'
    
    def test_multiple_dots_in_filename(self, test_data_dir):
        """Test filename with multiple dots."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        img = Image.new('RGB', (100, 100))
        dotted_path = img_dir / 'file.name.with.dots.png'
        img.save(dotted_path)
        
        FileValidator.validate_file_exists(dotted_path)  # Should not raise
        assert dotted_path.suffix == '.png'


class TestEmptyAndBoundaryInputs:
    """Test empty inputs and boundary conditions."""
    
    def test_empty_directory(self, test_data_dir):
        """Test processing empty directory."""
        empty_dir = test_data_dir / 'empty'
        empty_dir.mkdir()
        
        # Should be valid directory
        FileValidator.validate_directory(empty_dir)
    
    def test_minimum_size_image(self, test_data_dir):
        """Test image at minimum allowed size (10x10)."""
        from utils.validators import ImageValidator
        
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        # Create 10x10 image (minimum allowed)
        img = Image.new('RGB', (10, 10))
        min_path = img_dir / 'min_size.png'
        img.save(min_path)
        
        # Should pass validation
        width, height, mode = ImageValidator.validate_image(min_path)
        assert width == 10
        assert height == 10
    
    def test_non_square_image(self, test_data_dir):
        """Test rectangular (non-square) image."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        # Wide image
        img = Image.new('RGB', (1024, 100))
        wide_path = img_dir / 'wide.png'
        img.save(wide_path)
        
        preprocessor = ImagePreprocessor(target_size=(256, 256))
        tensor = preprocessor.preprocess(wide_path)
        
        # Should resize to square
        assert tensor.shape == (3, 256, 256)
    
    def test_very_thin_image(self, test_data_dir):
        """Test very thin image (edge case)."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        # 1000x10 image (ratio 100:1)
        img = Image.new('RGB', (1000, 10))
        thin_path = img_dir / 'thin.png'
        img.save(thin_path)
        
        preprocessor = ImagePreprocessor()
        tensor = preprocessor.preprocess(thin_path)
        assert tensor.shape == (3, 256, 256)


class TestImageFormatCornerCases:
    """Test various image formats and modes."""
    
    def test_rgba_image_with_transparency(self, test_data_dir):
        """Test RGBA image with alpha channel."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        # Create RGBA image
        img = Image.new('RGBA', (256, 256), (255, 0, 0, 128))
        rgba_path = img_dir / 'transparent.png'
        img.save(rgba_path)
        
        preprocessor = ImagePreprocessor()
        loaded = preprocessor.load_image(rgba_path)
        
        # Should convert to RGB
        assert loaded.mode == 'RGB'
    
    def test_palette_mode_image(self, test_data_dir):
        """Test palette mode (indexed color) image."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        # Create palette image
        img = Image.new('P', (256, 256))
        palette_path = img_dir / 'palette.png'
        img.save(palette_path)
        
        preprocessor = ImagePreprocessor()
        loaded = preprocessor.load_image(palette_path)
        
        # Should convert to RGB
        assert loaded.mode == 'RGB'
    
    def test_1bit_monochrome_image(self, test_data_dir):
        """Test 1-bit monochrome image."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        # Create 1-bit image
        img = Image.new('1', (256, 256))
        mono_path = img_dir / 'monochrome.png'
        img.save(mono_path)
        
        preprocessor = ImagePreprocessor()
        loaded = preprocessor.load_image(mono_path)
        
        # Should convert to RGB
        assert loaded.mode == 'RGB'


class TestConfigCornerCases:
    """Test configuration edge cases."""
    
    def test_config_with_extra_keys(self, test_data_dir):
        """Test that extra/unknown keys don't break validation."""
        config = {
            'processing': {'max_sequence_length': 128},
            'paths': {'raw_data': '.', 'interim_data': '.', 'processed_data': '.'},
            'logging': {'level': 'INFO', 'log_file': 'test.log'},
            'unknown_section': {'some_key': 'some_value'},  # Extra section
            'future_feature': True  # Extra key
        }
        
        config_path = test_data_dir / 'extra_keys.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        # Should still validate (extra keys ignored)
        from utils.validators import ConfigValidator
        validated = ConfigValidator.validate_config(config_path)
        assert validated is not None
    
    def test_config_with_comments(self, test_data_dir):
        """Test YAML config with comments."""
        config_text = """
# This is a comment
processing:
  max_sequence_length: 128  # Inline comment
  
# Another section
paths:
  raw_data: "."
  interim_data: "."
  processed_data: "."
  
logging:
  level: "INFO"
  log_file: "test.log"
"""
        config_path = test_data_dir / 'commented.yaml'
        with open(config_path, 'w') as f:
            f.write(config_text)
        
        from utils.validators import ConfigValidator
        validated = ConfigValidator.validate_config(config_path)
        assert validated['processing']['max_sequence_length'] == 128
    
    def test_zero_sequence_length(self, test_data_dir):
        """Test invalid config with zero sequence length."""
        config = {
            'processing': {'max_sequence_length': 0},  # Invalid!
            'paths': {'raw_data': '.', 'interim_data': '.', 'processed_data': '.'},
            'logging': {'level': 'INFO', 'log_file': 'test.log'}
        }
        
        config_path = test_data_dir / 'zero_seq.yaml'
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
        
        from utils.validators import ConfigValidator
        with pytest.raises(ValidationError) as exc_info:
            ConfigValidator.validate_config(config_path)
        
        assert "CONFIG_INVALID_VALUE" in str(exc_info.value)


class TestBatchProcessingCornerCases:
    """Test batch processing edge cases."""
    
    def test_batch_with_single_image(self, sample_image):
        """Test batch processing with only one image."""
        preprocessor = ImagePreprocessor()
        batch = preprocessor.preprocess_batch([sample_image])
        
        assert batch.shape == (1, 3, 256, 256)
    
    def test_batch_with_mixed_sizes(self, test_data_dir):
        """Test batch with images of different sizes."""
        img_dir = test_data_dir / 'images'
        img_dir.mkdir(exist_ok=True)
        
        # Create images of different sizes
        img1 = Image.new('RGB', (100, 100))
        img2 = Image.new('RGB', (500, 300))
        img3 = Image.new('RGB', (1024, 768))
        
        paths = []
        for i, img in enumerate([img1, img2, img3], 1):
            path = img_dir / f'img{i}.png'
            img.save(path)
            paths.append(path)
        
        preprocessor = ImagePreprocessor(target_size=(256, 256))
        batch = preprocessor.preprocess_batch(paths)
        
        # All should be resized to same size
        assert batch.shape == (3, 3, 256, 256)


class TestPathCornerCases:
    """Test path handling edge cases."""
    
    def test_relative_path(self, test_data_dir):
        """Test that relative paths work."""
        import os
        original_cwd = os.getcwd()
        
        try:
            os.chdir(test_data_dir)
            
            img_dir = Path('images')
            img_dir.mkdir(exist_ok=True)
            
            img = Image.new('RGB', (256, 256))
            rel_path = img_dir / 'relative.png'
            img.save(rel_path)
            
            FileValidator.validate_file_exists(rel_path)
        finally:
            os.chdir(original_cwd)
    
    def test_symlink_to_image(self, test_data_dir, sample_image):
        """Test handling of symbolic links."""
        import os
        
        link_path = test_data_dir / 'link_to_image.png'
        
        try:
            os.symlink(sample_image, link_path)
            
            # Should be able to process symlink
            preprocessor = ImagePreprocessor()
            tensor = preprocessor.preprocess(link_path)
            assert tensor.shape == (3, 256, 256)
        except OSError:
            # Symlinks might not be supported on this system
            pytest.skip("Symlinks not supported")

"""Pytest configuration and shared fixtures."""
import pytest
from pathlib import Path
import tempfile
import shutil
from PIL import Image
import torch
import yaml


@pytest.fixture(scope="session")
def test_data_dir():
    """Create temporary directory for test data."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config(test_data_dir):
    """Create sample configuration file."""
    config = {
        'processing': {
            'max_sequence_length': 128,
            'normalization': {
                'target_range': [0, 1],
                'preserve_aspect_ratio': True
            },
            'filters': {
                'cut_colors': ["#FF00FF", "Magenta"],
                'crease_colors': ["#00FFFF", "Cyan"]
            }
        },
        'paths': {
            'raw_data': str(test_data_dir / 'raw'),
            'interim_data': str(test_data_dir / 'interim'),
            'processed_data': str(test_data_dir / 'processed')
        },
        'inkscape': {
            'min_version': "1.2"
        },
        'logging': {
            'level': "INFO",
            'log_file': str(test_data_dir / "test.log")
        },
        'raster': {
            'enabled': True,
            'latent_dim': 512,
            'image_size': [256, 256],
            'device': 'cpu',
            'pretrained': True
        }
    }
    
    config_path = test_data_dir / 'config.yaml'
    with open(config_path, 'w') as f:
        yaml.dump(config, f)
    
    yield config_path


@pytest.fixture
def sample_image(test_data_dir):
    """Create sample test image."""
    img_dir = test_data_dir / 'images'
    img_dir.mkdir(exist_ok=True)
    
    # Create 256x256 RGB image
    img = Image.new('RGB', (256, 256))
    pixels = img.load()
    for i in range(256):
        for j in range(256):
            pixels[i, j] = (i, j, (i + j) // 2)
    
    img_path = img_dir / 'test_image.png'
    img.save(img_path)
    
    yield img_path


@pytest.fixture
def sample_small_image(test_data_dir):
    """Create small test image (edge case)."""
    img_dir = test_data_dir / 'images'
    img_dir.mkdir(exist_ok=True)
    
    img = Image.new('RGB', (50, 50), color='blue')
    img_path = img_dir / 'small_image.png'
    img.save(img_path)
    
    yield img_path


@pytest.fixture
def sample_grayscale_image(test_data_dir):
    """Create grayscale test image."""
    img_dir = test_data_dir / 'images'
    img_dir.mkdir(exist_ok=True)
    
    img = Image.new('L', (256, 256))
    pixels = img.load()
    for i in range(256):
        for j in range(256):
            pixels[i, j] = (i + j) // 2
    
    img_path = img_dir / 'gray_image.png'
    img.save(img_path)
    
    yield img_path


@pytest.fixture
def sample_tensor():
    """Create sample tensor for testing."""
    return torch.randn(128, 14)


@pytest.fixture
def invalid_config(test_data_dir):
    """Create invalid configuration file for testing error handling."""
    config_path = test_data_dir / 'invalid_config.yaml'
    with open(config_path, 'w') as f:
        f.write("invalid: {yaml: [syntax")
    
    yield config_path


@pytest.fixture
def corrupted_image(test_data_dir):
    """Create corrupted image file."""
    img_dir = test_data_dir / 'images'
    img_dir.mkdir(exist_ok=True)
    
    img_path = img_dir / 'corrupted.png'
    with open(img_path, 'w') as f:
        f.write(f"This is not an image file!")
    
    yield img_path

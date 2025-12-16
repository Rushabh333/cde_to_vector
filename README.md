# CDR-to-Vector Dual-Stream Pipeline

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Tests](https://img.shields.io/badge/tests-43%20passing-brightgreen.svg)]()
[![Coverage](https://img.shields.io/badge/coverage-89%25-green.svg)]()

> **A production-ready dual-stream ETL pipeline for converting both vector files (.cdr, .svg) and raster images (.png, .jpg) into normalized tensor representations for machine learning applications.**

Designed for vector-latent diffusion models, this pipeline provides robust processing of both vector graphics and raster images with comprehensive error handling, validation, and testing infrastructure.

---

## 🎯 Overview

This pipeline implements a complete Extract-Transform-Load (ETL) workflow that bridges proprietary design formats to machine learning-ready tensors. It supports two parallel processing streams:

- **Vector Stream**: Converts CorelDRAW (.cdr), SVG, and other vector formats to normalized Bézier curve representations
- **Raster Stream**: Encodes PNG, JPG, and other image formats to latent embeddings using pretrained ResNet-50

### Key Capabilities

- ✅ **Dual-Stream Architecture**: Automatic file type detection and routing
- ✅ **Production-Grade Robustness**: Comprehensive error handling with actionable messages
- ✅ **Extensive Testing**: 43 unit tests covering edge cases and corner cases
- ✅ **Flexible Configuration**: YAML-based settings for all pipeline parameters
- ✅ **Format Agnostic**: Supports all major vector and raster formats
- ✅ **Batch Processing**: Efficient processing of mixed file types
- ✅ **ML-Ready Outputs**: Fixed-dimensional PyTorch tensors

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Rushabh333/cde_to_vector.git
cd cde_to_vector

# Install dependencies
pip install -r requirements.txt

# Verify installation
python demo_pipeline.py  # Test vector processing
python demo_raster.py    # Test raster processing
```

### Basic Usage

```python
from pathlib import Path
from pipeline.unified_pipeline import UnifiedPipeline

# Initialize pipeline
pipeline = UnifiedPipeline(Path('config.yaml'))

# Process any file (auto-detects type)
output = pipeline.process(Path('design.cdr'))  # Vector → (128, 14)
output = pipeline.process(Path('photo.png'))   # Raster → (1, 512)

# Process entire directory
results = pipeline.process_directory(
    input_dir=Path('./data/mixed'),
    output_dir=Path('./output')
)
```

### Command Line

```bash
# Process single file
python run_unified_pipeline.py --input_file image.png

# Process directory with mixed file types
python run_unified_pipeline.py --input_dir ./data --output_dir ./output

# Vector-only processing (legacy)
python run_pipeline.py --input_dir ./cdr_files --seq_len 128
```

---

## ✨ Features

### Dual-Stream Processing

**Vector Stream** (`.cdr`, `.svg`, `.ai`, `.eps`):
- Headless Inkscape extraction preserving affine transformations
- Semantic path filtering (cut lines, crease lines)
- Geometric canonicalization to cubic Bézier curves
- Scale-invariant normalization to [0,1]² unit square
- Output: `(N, 14)` tensor per file

**Raster Stream** (`.png`, `.jpg`, `.bmp`, `.tiff`):
- Pretrained ResNet-50 feature extraction
- Automatic RGB conversion from any color mode
- Configurable image resizing (default 256×256)
- GPU/CPU support for acceleration
- Output: `(1, 512)` latent embedding per image

### Production Robustness

- **Comprehensive Validation**: Config schema, file formats, image dimensions
- **Actionable Error Messages**: Error codes with recovery suggestions
- **Corner Case Handling**: Unicode filenames, extreme aspect ratios, all PIL modes
- **Test Coverage**: 89% on validators, 90% on image preprocessing
- **Batch Processing**: Efficient handling of large datasets

### Configuration

Full YAML-based configuration with sensible defaults:

```yaml
# Processing settings
processing:
  max_sequence_length: 128
  normalization:
    target_range: [0, 1]
    preserve_aspect_ratio: true

# Raster processing (NEW)
raster:
  enabled: true
  encoder_type: "resnet"
  pretrained: true
  latent_dim: 512
  image_size: [256, 256]
  device: "cpu"  # or "cuda" for GPU

# Paths
paths:
  raw_data: "./data/raw"
  processed_data: "./data/processed"
```

---

## 📊 Architecture

### Data Flow

```
Input Files
    │
    ├── .cdr/.svg (Vector) ──┐
    │                        │
    │   Inkscape Extraction  │
    │   Semantic Filtering   │
    │   Bézier Conversion    ├──► Unified Output
    │   Normalization        │    (.pt tensors)
    │                        │
    └── .png/.jpg (Raster) ──┤
                             │
        Image Preprocessing  │
        ResNet-50 Encoding  │
        Latent Projection   │
                            │
```

### Project Structure

```
cde_to_vector/
├── src/
│   ├── extractors/      # Vector: CDR→SVG extraction
│   ├── geometry/        # Vector: Bézier conversion
│   ├── raster/          # Raster: Image processing
│   ├── pipeline/        # Unified routing & utilities
│   ├── tensor/          # Tensor serialization
│   └── utils/           # Validation & logging
├── tests/               # 43 unit tests
├── run_unified_pipeline.py  # Main CLI
├── config.yaml          # Configuration
└── README.md
```

---

## 🧪 Testing

Comprehensive test suite with 43 passing tests:

```bash
# Run all tests
pytest tests/ -v

# With coverage report
pytest tests/ --cov=src --cov-report=html

# Specific test categories
pytest tests/test_validators.py      # Validation tests
pytest tests/test_corner_cases.py    # Edge case tests
pytest tests/test_image_preprocessor.py  # Raster tests
```

**Test Coverage:**
- ✅ Config validation (schema, missing keys, invalid values)
- ✅ File validation (existence, formats, permissions)
- ✅ Image validation (dimensions, corruption, formats)
- ✅ Unicode filenames, special characters
- ✅ Extreme aspect ratios (100:1, 1:100)
- ✅ All PIL image modes (RGB, RGBA, L, P, 1)
- ✅ Boundary conditions (10×10 minimum size)
- ✅ Batch processing edge cases

---

## 📖 Documentation

- **[README.md](README.md)** - This file (complete usage guide)
- **[CITATIONS.md](CITATIONS.md)** - Academic references for research
- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)** - Detailed code organization
- **[CORNER_CASES_REPORT.md](CORNER_CASES_REPORT.md)** - Edge case testing results

---

## 🔧 Advanced Usage

### Custom Raster Processing

```python
from raster.raster_pipeline import RasterPipeline

# Initialize with custom settings
pipeline = RasterPipeline(
    latent_dim=512,
    image_size=(256, 256),
    device='cuda',  # Use GPU
    pretrained=True
)

# Process single image
pipeline.process_file('photo.jpg', 'output.pt')

# Batch processing
images = [Path(f) for f in glob('*.png')]
pipeline.process_batch(images, Path('outputs/'))
```

### Latent Standardization

```python
from pipeline.latent_standardizer import LatentStandardizer
import torch

# Standardize different formats to common dimension
standardizer = LatentStandardizer(target_dim=512)

# Convert vector latent (128, 14) → (512,)
vector_latent = torch.load('design.pt')
std_vector = standardizer.standardize(vector_latent, 'vector')

# Convert raster latent (1, 512) → (512,)
raster_latent = torch.load('photo.pt')
std_raster = standardizer.standardize(raster_latent, 'raster')

# Now both have shape (512,) for downstream models
```

### Error Handling

```python
from utils.validators import ValidationError

try:
    pipeline.process(input_file)
except ValidationError as e:
    # Access structured error information
    print(f"Code: {e.error_code}")
    print(f"Message: {e.message}")
    print(f"Suggestion: {e.suggestion}")
    
    # Programmatic error handling
    if e.error_code == "IMAGE_TOO_LARGE":
        # Resize and retry
        resize_image(input_file)
        pipeline.process(input_file)
```

---

## 🔬 Technical Details

### Vector Pipeline

**Mathematical Canonicalization:**

All primitives converted to uniform cubic Bézier curves:

| Primitive | Conversion Method |
|-----------|-------------------|
| Line | $P_0=L_0, P_1=\frac{2L_0+L_1}{3}, P_2=\frac{L_0+2L_1}{3}, P_3=L_1$ |
| Quadratic | Degree elevation (3 → 4 control points) |
| Arc | Split strategy where $\theta \leq 90°$ |

**Normalization:**
$$V_{norm} = \frac{V - \mu}{2s} + 0.5$$

Where $\mu$ = centroid, $s$ = max dimension

**Output Format:** $\mathbf{X} \in \mathbb{R}^{N_{max} \times 14}$

Feature vector per curve:
```
[x0, y0, x1, y1, x2, y2, x3, y3, r, g, b, α, type, pad]
```

### Raster Pipeline

**Architecture:**
1. **Preprocessing**: Resize to 256×256, ImageNet normalization
2. **Feature Extraction**: ResNet-50 (pretrained) → 2048D features
3. **Projection**: Linear(2048 → 1024) → ReLU → Linear(1024 → 512)

**Performance:**
- First run: ~30s (model download)
- Subsequent: <1s per image
- GPU acceleration supported
- Batch processing optimized

---

## 🛠️ Configuration Reference

### Complete Config Options

```yaml
# Processing parameters
processing:
  max_sequence_length: 128
  normalization:
    target_range: [0, 1]
    preserve_aspect_ratio: true
  filters:
    cut_colors: ["#FF00FF", "Magenta"]
    crease_colors: ["#00FFFF", "Cyan"]

# File paths
paths:
  raw_data: "./data/raw"
  interim_data: "./data/interim"
  processed_data: "./data/processed"

# Inkscape settings
inkscape:
  min_version: "1.2"
  export_type: "svg"
  export_plain_svg: true

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  log_file: "./pipeline.log"

# Raster processing
raster:
  enabled: true
  encoder_type: "resnet"
  pretrained: true
  latent_dim: 512
  image_size: [256, 256]
  device: "cpu"  # or "cuda"

# Unified pipeline
unified:
  mode: "independent"
  auto_detect_type: true
  supported_vector_formats: [".cdr", ".svg", ".ai", ".eps"]
  supported_raster_formats: [".png", ".jpg", ".jpeg", ".bmp", ".tiff"]
```

---

## 🐛 Troubleshooting

### Common Issues

**Issue**: `Inkscape not found`
```bash
# Install Inkscape
brew install inkscape  # macOS
sudo apt-get install inkscape  # Linux
```

**Issue**: `CUDA out of memory`
```yaml
# Use CPU instead in config.yaml
raster:
  device: "cpu"
```

**Issue**: `Image too large` error
```yaml
# Adjust max size in code or resize before processing
# Default limit: 50MB
```

**Issue**: `[CONFIG_MISSING_KEY]` error
- Check config.yaml has all required sections
- See error suggestion for specific missing key

---

## 📝 Examples

### Example 1: Process Mixed Design Files

```python
from pathlib import Path
from pipeline.unified_pipeline import UnifiedPipeline

# Initialize
pipeline = UnifiedPipeline(Path('config.yaml'))

# Process directory with both vectors and rasters
results = pipeline.process_directory(
    input_dir=Path('./designs'),
    output_dir=Path('./tensors')
)

print(f"Vector files: {len(results['vector'])}")
print(f"Raster files: {len(results['raster'])}")
```

### Example 2: Batch Processing with Progress

```python
from pathlib import Path
from tqdm import tqdm
from pipeline.unified_pipeline import UnifiedPipeline

pipeline = UnifiedPipeline(Path('config.yaml'))

files = list(Path('./data').glob('*'))
for file in tqdm(files, desc="Processing"):
    try:
        output = pipeline.process(file)
        print(f"✓ {file.name} → {output}")
    except Exception as e:
        print(f"✗ {file.name}: {e}")
```

---

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Write tests** for new functionality
4. **Ensure** all tests pass (`pytest tests/`)
5. **Commit** your changes (`git commit -m 'Add amazing feature'`)
6. **Push** to the branch (`git push origin feature/amazing-feature`)
7. **Open** a Pull Request

### Development Setup

```bash
# Clone repository
git clone https://github.com/Rushabh333/cde_to_vector.git
cd cde_to_vector

# Install development dependencies
pip install -r requirements.txt
pip install pytest pytest-cov

# Run tests
pytest tests/ -v --cov=src
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Rushabh Lodha**

- GitHub: [@Rushabh333](https://github.com/Rushabh333)

---

## 🙏 Acknowledgments

- **Inkscape Project** for the powerful vector graphics toolkit
- **PyTorch Team** for the deep learning framework
- **PIL/Pillow** for comprehensive image processing
- Academic references detailed in [CITATIONS.md](CITATIONS.md)

---

## 📊 Project Stats

| Metric | Value |
|--------|-------|
| **Lines of Code** | ~1,500 (production) |
| **Test Coverage** | 89% (validators), 90% (image preprocessing) |
| **Tests** | 43 passing |
| **Supported Formats** | 9 (vector + raster) |
| **Dependencies** | 11 core libraries |
| **Python Version** | 3.8+ |

---

## 🗺️ Roadmap

### Completed ✅
- [x] Vector pipeline (CDR, SVG support)
- [x] Raster pipeline (ResNet-50 encoder)
- [x] Unified interface with auto-detection
- [x] Comprehensive validation framework
- [x] Production-grade error handling
- [x] Extensive test suite (43 tests)
- [x] Corner case coverage

### Future Enhancements 🔮
- [ ] Lightweight fusion module (cross-attention between modalities)
- [ ] Alternative encoders (CLIP, DINOv2)
- [ ] GPU batch processing optimization
- [ ] Fine-tuning capabilities for domain-specific data
- [ ] WebUI for interactive processing
- [ ] Docker containerization
- [ ] CI/CD with GitHub Actions

---

## 📞 Support

For questions, issues, or suggestions:
- **Issues**: [GitHub Issues](https://github.com/Rushabh333/cde_to_vector/issues)
- **Discussions**: [GitHub Discussions](https://github.com/Rushabh333/cde_to_vector/discussions)

---

**⭐ If you find this project useful, please consider giving it a star!**

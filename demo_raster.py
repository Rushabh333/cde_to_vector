#!/usr/bin/env python3
"""Demo script for raster image processing."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from raster.raster_pipeline import RasterPipeline
from PIL import Image
import torch


def create_sample_image(output_path: Path):
    """Create a simple test image."""
    print("Creating sample test image...")
    
    # Create a 256x256 gradient image
    img = Image.new('RGB', (256, 256))
    pixels = img.load()
    
    for i in range(256):
        for j in range(256):
            # Create gradient pattern
            r = i
            g = j
            b = (i + j) // 2
            pixels[i, j] = (r, g, b)
    
    img.save(output_path)
    print(f"  ✓ Saved to: {output_path}")
    return output_path


def main():
    """Run the raster processing demo."""
    print("="*60)
    print("Raster Image Processing Demo")
    print("="*60)
    print()
    
    # Create directories
    data_dir = Path('./data')
    raw_dir = data_dir / 'raw'
    processed_dir = data_dir / 'processed'
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Create sample image
    sample_image = raw_dir / 'sample_gradient.png'
    if not sample_image.exists():
        create_sample_image(sample_image)
    else:
        print(f"Using existing sample image: {sample_image}")
    
    print()
    
    # Initialize pipeline
    print("Initializing raster pipeline...")
    pipeline = RasterPipeline(
        latent_dim=512,
        image_size=(256, 256),
        device='cpu',
        pretrained=True
    )
    print()
    
    # Process image
    output_path = processed_dir / 'sample_gradient.pt'
    print(f"Processing: {sample_image.name}")
    print()
    
    try:
        result_path = pipeline.process_file(sample_image, output_path)
        print()
        print("="*60)
        print("✓ Processing Complete!")
        print("="*60)
        print()
        
        # Load and inspect the output
        latent = torch.load(result_path)
        
        print("Output tensor information:")
        print(f"  File: {result_path}")
        print(f"  Shape: {latent.shape}")
        print(f"  Dtype: {latent.dtype}")
        print(f"  Device: {latent.device}")
        print(f"  Value range: [{latent.min():.4f}, {latent.max():.4f}]")
        print(f"  Mean: {latent.mean():.4f}")
        print(f"  Std: {latent.std():.4f}")
        print()
        
        print("Sample values (first 10 elements):")
        print(f"  {latent.flatten()[:10].tolist()}")
        print()
        
        print("✓ Demo completed successfully!")
        print(f"\nYou can now use this pipeline to process your own images:")
        print(f"  from raster.raster_pipeline import RasterPipeline")
        print(f"  pipeline = RasterPipeline()")
        print(f"  pipeline.process_file('your_image.png', 'output.pt')")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

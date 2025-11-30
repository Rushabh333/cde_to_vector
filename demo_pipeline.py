#!/usr/bin/env python3
"""Demo script to test the pipeline with a simple SVG file."""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from extractors.filter import SVGFilter
from geometry.bezier_converter import BezierConverter
from geometry.normalizer import Normalizer
from tensor.serializer import TensorSerializer
from utils.logger import setup_logger
import numpy as np


def create_sample_svg():
    """Create a simple SVG file for testing."""
    svg_content = """<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100" viewBox="0 0 100 100">
  <!-- Cut lines (Magenta) -->
  <path d="M 10,10 L 90,10 L 90,90 L 10,90 Z" stroke="#FF00FF" fill="none" stroke-width="1"/>
  
  <!-- Crease line (Cyan) -->
  <path d="M 10,50 L 90,50" stroke="#00FFFF" fill="none" stroke-width="1"/>
  
  <!-- Curved path (Magenta) -->
  <path d="M 30,30 Q 50,10 70,30" stroke="Magenta" fill="none" stroke-width="1"/>
</svg>"""
    
    svg_path = Path('./data/interim/sample_test.svg')
    svg_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(svg_path, 'w') as f:
        f.write(svg_content)
    
    return svg_path


def main():
    """Run demo pipeline."""
    # Setup logger
    logger = setup_logger('Demo_Pipeline', level='INFO')
    
    logger.info("="*60)
    logger.info("CDR to Vector Pipeline - DEMO MODE")
    logger.info("="*60)
    
    # Create sample SVG
    logger.info("\n1. Creating sample SVG file...")
    svg_path = create_sample_svg()
    logger.info(f"   ✓ Created: {svg_path}")
    
    # Filter paths
    logger.info("\n2. Filtering semantic paths...")
    filter = SVGFilter(
        cut_colors=["#FF00FF", "Magenta"],
        crease_colors=["#00FFFF", "Cyan"],
        logger=logger
    )
    paths = filter.filter(svg_path)
    
    logger.info(f"   ✓ Found {len(paths['cut'])} cut paths")
    logger.info(f"   ✓ Found {len(paths['crease'])} crease paths")
    
    # Convert to Bézier curves
    logger.info("\n3. Converting to canonical Bézier curves...")
    converter = BezierConverter(logger=logger)
    
    all_control_points = []
    for path_type in ['cut', 'crease']:
        for path_info in paths[path_type]:
            try:
                cubics = converter.convert_path(path_info['data'])
                for cubic in cubics:
                    cp = converter.extract_control_points(cubic)
                    all_control_points.append(cp)
            except Exception as e:
                logger.warning(f"   ⚠ Failed to convert path: {e}")
    
    logger.info(f"   ✓ Converted to {len(all_control_points)} Bézier curves")
    
    # Normalize
    logger.info("\n4. Normalizing coordinates to [0,1]²...")
    normalizer = Normalizer(
        target_range=(0.0, 1.0),
        preserve_aspect_ratio=True,
        logger=logger
    )
    
    control_points_array = np.array(all_control_points)
    normalized_points = normalizer.normalize(control_points_array)
    normalized_list = [normalized_points[i] for i in range(len(normalized_points))]
    
    logger.info(f"   ✓ Normalized {len(normalized_list)} curves")
    
    # Serialize to tensor
    logger.info("\n5. Serializing to tensor...")
    serializer = TensorSerializer(max_sequence_length=128, logger=logger)
    
    output_path = Path('./data/processed/sample_test.pt')
    serializer.serialize(normalized_list, output_path, format='pt')
    
    logger.info(f"   ✓ Saved tensor to: {output_path}")
    
    # Load and display info
    logger.info("\n6. Verifying output...")
    import torch
    tensor = torch.load(output_path)
    
    logger.info(f"   ✓ Tensor shape: {tensor.shape}")
    logger.info(f"   ✓ Data type: {tensor.dtype}")
    logger.info(f"   ✓ Value range: [{tensor.min():.3f}, {tensor.max():.3f}]")
    logger.info(f"   ✓ Non-zero rows: {(tensor.abs().sum(dim=1) > 0).sum().item()}")
    
    logger.info("\n" + "="*60)
    logger.info("✅ DEMO COMPLETE - Pipeline is working correctly!")
    logger.info("="*60)
    logger.info("\nTo use with real .cdr files:")
    logger.info("  1. Place .cdr files in ./data/raw/")
    logger.info("  2. Run: python run_pipeline.py")
    logger.info("="*60)


if __name__ == '__main__':
    main()

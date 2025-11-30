#!/usr/bin/env python3
"""Main pipeline orchestrator for CDR to Vector conversion."""
import argparse
import yaml
from pathlib import Path
from typing import List
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from extractors.inkscape_wrapper import InkscapeExtractor
from extractors.filter import SVGFilter
from geometry.bezier_converter import BezierConverter
from geometry.normalizer import Normalizer
from tensor.serializer import TensorSerializer
from utils.logger import setup_logger


class Pipeline:
    """Main ETL pipeline orchestrator."""
    
    def __init__(self, config_path: Path):
        """
        Initialize the pipeline with configuration.
        
        Args:
            config_path: Path to config.yaml file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Setup logger
        self.logger = setup_logger(
            'CDR_Pipeline',
            log_file=self.config['logging']['log_file'],
            level=self.config['logging']['level']
        )
        
        self.logger.info("="*60)
        self.logger.info("CDR to Vector Pipeline - Starting")
        self.logger.info("="*60)
        
        # Initialize components
        self.extractor = InkscapeExtractor(
            min_version=self.config['inkscape']['min_version'],
            logger=self.logger
        )
        
        self.filter = SVGFilter(
            cut_colors=self.config['processing']['filters']['cut_colors'],
            crease_colors=self.config['processing']['filters']['crease_colors'],
            logger=self.logger
        )
        
        self.converter = BezierConverter(logger=self.logger)
        
        self.normalizer = Normalizer(
            target_range=tuple(self.config['processing']['normalization']['target_range']),
            preserve_aspect_ratio=self.config['processing']['normalization']['preserve_aspect_ratio'],
            logger=self.logger
        )
        
        self.serializer = TensorSerializer(
            max_sequence_length=self.config['processing']['max_sequence_length'],
            logger=self.logger
        )
    
    def process_file(self, cdr_path: Path, output_dir: Path) -> Path:
        """
        Process a single CDR file through the pipeline.
        
        Args:
            cdr_path: Path to input .cdr file
            output_dir: Directory for output tensors
        
        Returns:
            Path to output tensor file
        """
        self.logger.info(f"\n{'='*60}")
        self.logger.info(f"Processing: {cdr_path.name}")
        self.logger.info(f"{'='*60}")
        
        # Step 1: Extract CDR to SVG
        interim_dir = Path(self.config['paths']['interim_data'])
        svg_path = interim_dir / f"{cdr_path.stem}.svg"
        
        try:
            svg_path = self.extractor.extract(cdr_path, svg_path)
        except Exception as e:
            self.logger.error(f"Extraction failed: {e}")
            raise
        
        # Step 2: Filter semantic paths
        try:
            paths = self.filter.filter(svg_path)
        except Exception as e:
            self.logger.error(f"Filtering failed: {e}")
            raise
        
        # Combine cut and crease paths (ignore 'other')
        relevant_paths = paths['cut'] + paths['crease']
        
        if not relevant_paths:
            self.logger.warning("No cut or crease paths found, using all paths")
            relevant_paths = paths['other']
        
        if not relevant_paths:
            self.logger.error("No paths found in SVG")
            raise ValueError("No paths to process")
        
        # Step 3: Convert to canonical Bézier curves
        all_control_points = []
        
        for path_info in relevant_paths:
            try:
                cubics = self.converter.convert_path(path_info['data'])
                for cubic in cubics:
                    cp = self.converter.extract_control_points(cubic)
                    all_control_points.append(cp)
            except Exception as e:
                self.logger.warning(f"Failed to convert path: {e}")
                continue
        
        if not all_control_points:
            self.logger.error("No valid Bézier curves generated")
            raise ValueError("No Bézier curves to serialize")
        
        self.logger.info(f"Converted to {len(all_control_points)} Bézier curves")
        
        # Step 4: Normalize coordinates
        import numpy as np
        control_points_array = np.array(all_control_points)
        normalized_points = self.normalizer.normalize(control_points_array)
        
        # Convert back to list
        normalized_list = [normalized_points[i] for i in range(len(normalized_points))]
        
        # Step 5: Serialize to tensor
        output_path = output_dir / f"{cdr_path.stem}.pt"
        
        try:
            output_path = self.serializer.serialize(
                normalized_list,
                output_path,
                format="pt"
            )
        except Exception as e:
            self.logger.error(f"Serialization failed: {e}")
            raise
        
        self.logger.info(f"✓ Successfully processed {cdr_path.name}")
        return output_path
    
    def run(self, input_dir: Path, output_dir: Path):
        """
        Run the pipeline on all CDR files in the input directory.
        
        Args:
            input_dir: Directory containing .cdr files
            output_dir: Directory for output tensors
        """
        # Find all CDR files
        cdr_files = list(input_dir.glob("*.cdr"))
        
        if not cdr_files:
            self.logger.warning(f"No .cdr files found in {input_dir}")
            return
        
        self.logger.info(f"Found {len(cdr_files)} CDR files to process")
        
        # Process each file
        successful = 0
        failed = 0
        
        for cdr_file in cdr_files:
            try:
                self.process_file(cdr_file, output_dir)
                successful += 1
            except Exception as e:
                self.logger.error(f"Failed to process {cdr_file.name}: {e}")
                failed += 1
                continue
        
        # Summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Pipeline Complete")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Successful: {successful}/{len(cdr_files)}")
        self.logger.info(f"Failed: {failed}/{len(cdr_files)}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CDR to Vector Latent Tensor Pipeline"
    )
    parser.add_argument(
        '--input_dir',
        type=Path,
        default=Path('./data/raw'),
        help='Directory containing .cdr files'
    )
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('./data/processed'),
        help='Directory for output tensors'
    )
    parser.add_argument(
        '--seq_len',
        type=int,
        default=128,
        help='Maximum sequence length'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('./config.yaml'),
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    
    # Validate input directory
    if not args.input_dir.exists():
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)
    
    # Validate config file
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Update config with command-line args
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    config['processing']['max_sequence_length'] = args.seq_len
    
    # Save updated config temporarily
    temp_config = Path('/tmp/pipeline_config.yaml')
    with open(temp_config, 'w') as f:
        yaml.dump(config, f)
    
    # Run pipeline
    try:
        pipeline = Pipeline(temp_config)
        pipeline.run(args.input_dir, args.output_dir)
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()

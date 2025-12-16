#!/usr/bin/env python3
"""Unified pipeline CLI for processing both vector and raster files."""
import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from pipeline.unified_pipeline import UnifiedPipeline


def main():
    """Main entry point for unified pipeline."""
    parser = argparse.ArgumentParser(
        description="Unified Pipeline: Process vector (.cdr, .svg) and raster (.png, .jpg) files"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        '--input_dir',
        type=Path,
        help='Directory containing input files (auto-detects types)'
    )
    input_group.add_argument(
        '--input_file',
        type=Path,
        help='Single file to process'
    )
    
    # Output options
    parser.add_argument(
        '--output_dir',
        type=Path,
        default=Path('./data/processed'),
        help='Directory for output tensors (default: ./data/processed)'
    )
    parser.add_argument(
        '--output_file',
        type=Path,
        help='Output file path (only with --input_file)'
    )
    
    # Processing options
    parser.add_argument(
        '--modality',
        type=str,
        choices=['vector', 'raster', 'auto'],
        default='auto',
        help='Force modality type (default: auto-detect)'
    )
    parser.add_argument(
        '--config',
        type=Path,
        default=Path('./config.yaml'),
        help='Path to configuration file (default: ./config.yaml)'
    )
    parser.add_argument(
        '--patterns',
        nargs='+',
        help='File patterns to match (e.g., *.png *.cdr)'
    )
    
    args = parser.parse_args()
    
    # Validate config
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    # Initialize pipeline
    try:
        print("Initializing unified pipeline...")
        pipeline = UnifiedPipeline(args.config)
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        sys.exit(1)
    
    # Process based on input type
    try:
        if args.input_file:
            # Single file processing
            if not args.input_file.exists():
                print(f"Error: Input file not found: {args.input_file}")
                sys.exit(1)
            
            modality = None if args.modality == 'auto' else args.modality
            
            output_path = pipeline.process(
                args.input_file,
                output_path=args.output_file,
                modality=modality
            )
            
            print(f"\n✓ Successfully processed: {args.input_file.name}")
            print(f"  Output: {output_path}")
        
        elif args.input_dir:
            # Directory processing
            if not args.input_dir.exists():
                print(f"Error: Input directory not found: {args.input_dir}")
                sys.exit(1)
            
            results = pipeline.process_directory(
                args.input_dir,
                args.output_dir,
                file_patterns=args.patterns
            )
            
            print(f"\n✓ Processing complete!")
            print(f"  Vector files processed: {len(results['vector'])}")
            print(f"  Raster files processed: {len(results['raster'])}")
            print(f"  Output directory: {args.output_dir}")
    
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nPipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()

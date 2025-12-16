"""Unified pipeline with automatic routing for vector and raster inputs."""
from pathlib import Path
from typing import Optional, Literal, Union
import sys
import logging
import yaml

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from raster.raster_pipeline import RasterPipeline
from utils.validators import ConfigValidator, ValidationError


class UnifiedPipeline:
    """Unified pipeline supporting both vector and raster inputs."""
    
    # Supported file formats
    VECTOR_FORMATS = {'.cdr', '.svg', '.ai', '.eps'}
    RASTER_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    def __init__(self, config_path: Path):
        """
        Initialize the unified pipeline.
        
        Args:
            config_path: Path to configuration YAML file
            
        Raises:
            ValidationError: If config is invalid
        """
        # Validate and load configuration
        self.config = ConfigValidator.validate_config(config_path)
        
        # Setup logger
        from utils.logger import setup_logger
        self.logger = setup_logger(
            'Unified_Pipeline',
            log_file=self.config['logging']['log_file'],
            level=self.config['logging']['level']
        )
        
        self.logger.info("="*60)
        self.logger.info("Unified Pipeline - Initializing")
        self.logger.info("="*60)
        
        # Initialize vector pipeline (lazy loading)
        self._vector_pipeline = None
        
        # Initialize raster pipeline
        raster_config = self.config.get('raster', {})
        if raster_config.get('enabled', True):
            self.logger.info("Initializing raster processing pipeline")
            self.raster_pipeline = RasterPipeline(
                latent_dim=raster_config.get('latent_dim', 512),
                image_size=tuple(raster_config.get('image_size', [256, 256])),
                device=raster_config.get('device', 'cpu'),
                pretrained=raster_config.get('pretrained', True),
                logger=self.logger
            )
        else:
            self.raster_pipeline = None
            self.logger.info("Raster processing disabled in config")
    
    @property
    def vector_pipeline(self):
        """Lazy-load vector pipeline only when needed."""
        if self._vector_pipeline is None:
            self.logger.info("Initializing vector processing pipeline")
            # Import here to avoid circular dependency
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            from run_pipeline import Pipeline
            self._vector_pipeline = Pipeline(
                Path(__file__).parent.parent.parent / 'config.yaml'
            )
        return self._vector_pipeline
    
    def detect_modality(self, file_path: Path) -> Literal['vector', 'raster']:
        """
        Detect whether input is vector or raster based on extension.
        
        Args:
            file_path: Path to input file
            
        Returns:
            'vector' or 'raster'
            
        Raises:
            ValueError: If file format is not supported
        """
        ext = file_path.suffix.lower()
        
        if ext in self.VECTOR_FORMATS:
            return 'vector'
        elif ext in self.RASTER_FORMATS:
            return 'raster'
        else:
            raise ValueError(
                f"Unsupported file format: {ext}. "
                f"Supported: {self.VECTOR_FORMATS | self.RASTER_FORMATS}"
            )
    
    def process(
        self,
        input_path: Path,
        output_path: Optional[Path] = None,
        modality: Optional[Literal['vector', 'raster']] = None
    ) -> Path:
        """
        Process a single file (auto-detects type or uses specified modality).
        
        Args:
            input_path: Path to input file
            output_path: Optional path for output (auto-generated if None)
            modality: Optional modality override ('vector' or 'raster')
            
        Returns:
            Path to output tensor file
        """
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")
        
        # Detect modality if not specified
        if modality is None:
            modality = self.detect_modality(input_path)
        
        self.logger.info(f"Detected modality: {modality} for {input_path.name}")
        
        # Generate output path if not provided
        if output_path is None:
            output_dir = Path(self.config['paths']['processed_data'])
            output_path = output_dir / f"{input_path.stem}.pt"
        
        # Route to appropriate pipeline
        if modality == 'vector':
            if self._vector_pipeline is None:
                self.logger.info("Loading vector pipeline...")
            return self.vector_pipeline.process_file(
                input_path,
                output_path.parent
            )
        
        elif modality == 'raster':
            if self.raster_pipeline is None:
                raise RuntimeError(
                    "Raster processing is disabled in configuration"
                )
            return self.raster_pipeline.process_file(
                input_path,
                output_path
            )
        
        else:
            raise ValueError(f"Unknown modality: {modality}")
    
    def process_directory(
        self,
        input_dir: Path,
        output_dir: Path,
        file_patterns: Optional[list[str]] = None
    ) -> dict[str, list[Path]]:
        """
        Process all supported files in a directory.
        
        Args:
            input_dir: Directory containing input files
            output_dir: Directory for output tensors
            file_patterns: Optional list of glob patterns (e.g., ['*.png', '*.cdr'])
            
        Returns:
            Dictionary with 'vector' and 'raster' keys, each listing output paths
        """
        if not input_dir.exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all supported files
        if file_patterns is None:
            all_exts = self.VECTOR_FORMATS | self.RASTER_FORMATS
            files = []
            for ext in all_exts:
                files.extend(input_dir.glob(f"*{ext}"))
        else:
            files = []
            for pattern in file_patterns:
                files.extend(input_dir.glob(pattern))
        
        self.logger.info(f"Found {len(files)} files to process in {input_dir}")
        
        # Process each file
        results = {'vector': [], 'raster': []}
        successful = 0
        failed = 0
        
        for file_path in files:
            try:
                modality = self.detect_modality(file_path)
                output_path = self.process(file_path)
                results[modality].append(output_path)
                successful += 1
            except Exception as e:
                self.logger.error(f"Failed to process {file_path.name}: {e}")
                failed += 1
                continue
        
        # Summary
        self.logger.info(f"\n{'='*60}")
        self.logger.info("Processing Complete")
        self.logger.info(f"{'='*60}")
        self.logger.info(f"Successful: {successful}/{len(files)}")
        self.logger.info(f"Failed: {failed}/{len(files)}")
        self.logger.info(f"Vector files: {len(results['vector'])}")
        self.logger.info(f"Raster files: {len(results['raster'])}")
        
        return results

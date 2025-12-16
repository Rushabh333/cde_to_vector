"""Input validation utilities for production robustness."""
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import yaml
from PIL import Image
import logging


class ValidationError(Exception):
    """Custom exception for validation errors with actionable messages."""
    
    def __init__(self, message: str, suggestion: str = None, error_code: str = None):
        self.message = message
        self.suggestion = suggestion
        self.error_code = error_code
        super().__init__(self.format_message())
    
    def format_message(self) -> str:
        """Format error message with code and suggestion."""
        parts = []
        if self.error_code:
            parts.append(f"[{self.error_code}]")
        parts.append(self.message)
        if self.suggestion:
            parts.append(f"\n💡 Suggestion: {self.suggestion}")
        return " ".join(parts)


class ConfigValidator:
    """Validate configuration files."""
    
    REQUIRED_KEYS = {
        'processing': ['max_sequence_length'],
        'paths': ['raw_data', 'interim_data', 'processed_data'],
        'logging': ['level', 'log_file']
    }
    
    OPTIONAL_KEYS = {
        'raster': ['enabled', 'latent_dim', 'image_size', 'device'],
        'unified': ['mode', 'auto_detect_type']
    }
    
    @staticmethod
    def validate_config(config_path: Path) -> Dict[str, Any]:
        """
        Validate configuration file.
        
        Args:
            config_path: Path to config YAML file
            
        Returns:
            Validated config dictionary
            
        Raises:
            ValidationError: If config is invalid
        """
        # Check file exists
        if not config_path.exists():
            raise ValidationError(
                f"Config file not found: {config_path}",
                suggestion=f"Create config.yaml or specify path with --config flag",
                error_code="CONFIG_NOT_FOUND"
            )
        
        # Parse YAML
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ValidationError(
                f"Invalid YAML syntax in {config_path}: {e}",
                suggestion="Check YAML formatting (indentation, colons, etc.)",
                error_code="CONFIG_INVALID_YAML"
            )
        
        if not isinstance(config, dict):
            raise ValidationError(
                "Config must be a YAML dictionary",
                suggestion="Ensure config starts with key-value pairs, not a list",
                error_code="CONFIG_INVALID_FORMAT"
            )
        
        # Validate required keys
        for section, keys in ConfigValidator.REQUIRED_KEYS.items():
            if section not in config:
                raise ValidationError(
                    f"Missing required section '{section}' in config",
                    suggestion=f"Add '{section}:' section with keys: {', '.join(keys)}",
                    error_code="CONFIG_MISSING_SECTION"
                )
            
            for key in keys:
                if key not in config[section]:
                    raise ValidationError(
                        f"Missing required key '{key}' in section '{section}'",
                        suggestion=f"Add '{key}:' under '{section}' section",
                        error_code="CONFIG_MISSING_KEY"
                    )
        
        # Validate value types
        if not isinstance(config['processing']['max_sequence_length'], int):
            raise ValidationError(
                "max_sequence_length must be an integer",
                suggestion="Set max_sequence_length: 128 (or other positive integer)",
                error_code="CONFIG_INVALID_TYPE"
            )
        
        if config['processing']['max_sequence_length'] <= 0:
            raise ValidationError(
                "max_sequence_length must be positive",
                suggestion="Use value like 128, 256, or 512",
                error_code="CONFIG_INVALID_VALUE"
            )
        
        return config


class FileValidator:
    """Validate input files."""
    
    VECTOR_FORMATS = {'.cdr', '.svg', '.ai', '.eps'}
    RASTER_FORMATS = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp'}
    
    @staticmethod
    def validate_file_exists(file_path: Path) -> None:
        """Check if file exists."""
        if not file_path.exists():
            raise ValidationError(
                f"File not found: {file_path}",
                suggestion=f"Check file path and ensure file exists",
                error_code="FILE_NOT_FOUND"
            )
        
        if not file_path.is_file():
            raise ValidationError(
                f"Path is not a file: {file_path}",
                suggestion="Provide path to a file, not a directory",
                error_code="NOT_A_FILE"
            )
    
    @staticmethod
    def validate_file_format(
        file_path: Path,
        allowed_formats: Optional[set] = None
    ) -> str:
        """
        Validate file format.
        
        Args:
            file_path: Path to file
            allowed_formats: Set of allowed extensions (e.g., {'.png', '.jpg'})
            
        Returns:
            File extension (lowercase)
            
        Raises:
            ValidationError: If format is invalid
        """
        ext = file_path.suffix.lower()
        
        if not ext:
            raise ValidationError(
                f"File has no extension: {file_path.name}",
                suggestion="Ensure file has proper extension (.png, .jpg, .cdr, etc.)",
                error_code="NO_FILE_EXTENSION"
            )
        
        if allowed_formats and ext not in allowed_formats:
            raise ValidationError(
                f"Unsupported file format '{ext}' for {file_path.name}",
                suggestion=f"Use one of: {', '.join(sorted(allowed_formats))}",
                error_code="UNSUPPORTED_FORMAT"
            )
        
        return ext
    
    @staticmethod
    def validate_directory(dir_path: Path, create: bool = False) -> None:
        """
        Validate directory exists or create it.
        
        Args:
            dir_path: Path to directory
            create: Whether to create if doesn't exist
            
        Raises:
            ValidationError: If directory invalid and create=False
        """
        if not dir_path.exists():
            if create:
                try:
                    dir_path.mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    raise ValidationError(
                        f"Cannot create directory {dir_path}: {e}",
                        suggestion="Check permissions and parent directory",
                        error_code="CANNOT_CREATE_DIR"
                    )
            else:
                raise ValidationError(
                    f"Directory not found: {dir_path}",
                    suggestion="Create directory or use --output_dir to specify path",
                    error_code="DIR_NOT_FOUND"
                )
        
        if not dir_path.is_dir():
            raise ValidationError(
                f"Path is not a directory: {dir_path}",
                suggestion="Provide path to a directory, not a file",
                error_code="NOT_A_DIRECTORY"
            )


class ImageValidator:
    """Validate image files."""
    
    MAX_DIMENSION = 10000  # pixels
    MIN_DIMENSION = 10     # pixels
    
    @staticmethod
    def validate_image(
        image_path: Path,
        max_size_mb: Optional[float] = None
    ) -> Tuple[int, int, str]:
        """
        Validate image file can be loaded and has reasonable dimensions.
        
        Args:
            image_path: Path to image file
            max_size_mb: Maximum file size in MB (optional)
            
        Returns:
            Tuple of (width, height, mode)
            
        Raises:
            ValidationError: If image is invalid
        """
        # Check file size
        if max_size_mb:
            size_mb = image_path.stat().st_size / (1024 * 1024)
            if size_mb > max_size_mb:
                raise ValidationError(
                    f"Image file too large: {size_mb:.1f}MB (max: {max_size_mb}MB)",
                    suggestion="Resize or compress image before processing",
                    error_code="IMAGE_TOO_LARGE"
                )
        
        # Try to open image
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                mode = img.mode
        except Exception as e:
            raise ValidationError(
                f"Cannot open image {image_path.name}: {e}",
                suggestion="Ensure file is a valid image (not corrupted)",
                error_code="IMAGE_CORRUPT"
            )
        
        # Validate dimensions
        if width < ImageValidator.MIN_DIMENSION or height < ImageValidator.MIN_DIMENSION:
            raise ValidationError(
                f"Image too small: {width}x{height} (min: {ImageValidator.MIN_DIMENSION}px)",
                suggestion="Use images at least 10x10 pixels",
                error_code="IMAGE_TOO_SMALL"
            )
        
        if width > ImageValidator.MAX_DIMENSION or height > ImageValidator.MAX_DIMENSION:
            raise ValidationError(
                f"Image too large: {width}x{height} (max: {ImageValidator.MAX_DIMENSION}px)",
                suggestion="Resize image to reasonable dimensions",
                error_code="IMAGE_DIMENSIONS_INVALID"
            )
        
        return width, height, mode


# Convenience function for quick validation
def validate_input(
    input_path: Path,
    modality: Optional[str] = None
) -> str:
    """
    Quick validation of input file.
    
    Args:
        input_path: Path to input file
        modality: Optional modality ('vector' or 'raster')
        
    Returns:
        Detected modality
        
    Raises:
        ValidationError: If input is invalid
    """
    FileValidator.validate_file_exists(input_path)
    
    ext = input_path.suffix.lower()
    
    # Determine modality
    if ext in FileValidator.VECTOR_FORMATS:
        detected_modality = 'vector'
    elif ext in FileValidator.RASTER_FORMATS:
        detected_modality = 'raster'
    else:
        all_formats = FileValidator.VECTOR_FORMATS | FileValidator.RASTER_FORMATS
        raise ValidationError(
            f"Unknown file format: {ext}",
            suggestion=f"Supported formats: {', '.join(sorted(all_formats))}",
            error_code="UNKNOWN_FORMAT"
        )
    
    # Check modality matches if specified
    if modality and modality != detected_modality:
        raise ValidationError(
            f"File {input_path.name} is {detected_modality} but {modality} was requested",
            suggestion="Let pipeline auto-detect or provide correct file type",
            error_code="MODALITY_MISMATCH"
        )
    
    # Additional validation for raster images
    if detected_modality == 'raster':
        ImageValidator.validate_image(input_path, max_size_mb=50)
    
    return detected_modality

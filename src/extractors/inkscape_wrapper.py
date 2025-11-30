"""Inkscape CLI wrapper for CDR to SVG conversion."""
import subprocess
import shutil
from pathlib import Path
from typing import Optional
import logging


class InkscapeExtractor:
    """Wrapper for Inkscape CLI to convert CDR files to SVG."""
    
    def __init__(self, min_version: str = "1.2", logger: Optional[logging.Logger] = None):
        """
        Initialize the Inkscape extractor.
        
        Args:
            min_version: Minimum required Inkscape version
            logger: Logger instance for output
        
        Raises:
            RuntimeError: If Inkscape is not found or version is incompatible
        """
        self.min_version = min_version
        self.logger = logger or logging.getLogger(__name__)
        
        # Validate Inkscape installation
        self._validate_inkscape()
    
    def _validate_inkscape(self):
        """Check if Inkscape is installed and meets version requirements."""
        if not shutil.which("inkscape"):
            raise RuntimeError(
                "Inkscape not found in system PATH. "
                "Please install Inkscape v1.2+ from https://inkscape.org/"
            )
        
        try:
            result = subprocess.run(
                ["inkscape", "--version"],
                capture_output=True,
                text=True,
                check=True
            )
            version_output = result.stdout.strip()
            self.logger.info(f"Found {version_output}")
            
            # Extract version number (e.g., "Inkscape 1.2.1 ...")
            version_parts = version_output.split()[1].split('.')
            major_minor = f"{version_parts[0]}.{version_parts[1]}"
            
            if float(major_minor) < float(self.min_version):
                raise RuntimeError(
                    f"Inkscape version {major_minor} found, but {self.min_version}+ is required"
                )
                
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to check Inkscape version: {e}")
    
    def extract(self, cdr_path: Path, output_path: Path) -> Path:
        """
        Convert a CDR file to SVG using Inkscape.
        
        Args:
            cdr_path: Path to input .cdr file
            output_path: Path for output .svg file
        
        Returns:
            Path to the generated SVG file
        
        Raises:
            FileNotFoundError: If CDR file doesn't exist
            RuntimeError: If conversion fails
        """
        if not cdr_path.exists():
            raise FileNotFoundError(f"CDR file not found: {cdr_path}")
        
        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Converting {cdr_path.name} to SVG...")
        
        try:
            # Run Inkscape conversion
            # --export-plain-svg: Preserve affine transformations
            # --export-type=svg: Output format
            subprocess.run(
                [
                    "inkscape",
                    "--export-type=svg",
                    "--export-plain-svg",
                    f"--export-filename={output_path}",
                    str(cdr_path)
                ],
                capture_output=True,
                text=True,
                check=True,
                timeout=60  # 60 second timeout per file
            )
            
            if not output_path.exists():
                raise RuntimeError(f"Conversion produced no output file: {output_path}")
            
            self.logger.info(f"Successfully converted to {output_path.name}")
            return output_path
            
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"Inkscape conversion timed out for {cdr_path.name}")
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr if e.stderr else str(e)
            raise RuntimeError(f"Inkscape conversion failed: {error_msg}")

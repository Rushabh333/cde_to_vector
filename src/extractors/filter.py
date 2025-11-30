"""SVG semantic filtering for cut/crease line identification."""
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging
import re


class SVGFilter:
    """Filter and parse SVG files to extract semantic path data."""
    
    # SVG namespace
    SVG_NS = {'svg': 'http://www.w3.org/2000/svg'}
    
    def __init__(
        self, 
        cut_colors: List[str] = None,
        crease_colors: List[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        """
        Initialize the SVG filter.
        
        Args:
            cut_colors: List of color values for cut lines (e.g., ["#FF00FF", "Magenta"])
            crease_colors: List of color values for crease lines (e.g., ["#00FFFF", "Cyan"])
            logger: Logger instance
        """
        self.cut_colors = set(c.lower() for c in (cut_colors or ["#ff00ff", "magenta"]))
        self.crease_colors = set(c.lower() for c in (crease_colors or ["#00ffff", "cyan"]))
        self.logger = logger or logging.getLogger(__name__)
    
    def _normalize_color(self, color: str) -> str:
        """Normalize color string to lowercase and remove whitespace."""
        if not color:
            return ""
        return color.strip().lower()
    
    def _extract_color_from_style(self, style: str) -> Optional[str]:
        """Extract stroke color from SVG style attribute."""
        if not style:
            return None
        
        # Match stroke:color or stroke:#hexcolor
        stroke_match = re.search(r'stroke\s*:\s*([^;]+)', style)
        if stroke_match:
            return self._normalize_color(stroke_match.group(1))
        return None
    
    def _get_element_color(self, element: ET.Element) -> Optional[str]:
        """Get the stroke color of an SVG element."""
        # Check stroke attribute
        stroke = element.get('stroke')
        if stroke:
            return self._normalize_color(stroke)
        
        # Check style attribute
        style = element.get('style')
        if style:
            return self._extract_color_from_style(style)
        
        return None
    
    def _classify_path(self, color: Optional[str]) -> str:
        """
        Classify a path based on its color.
        
        Args:
            color: Normalized color string
        
        Returns:
            'cut', 'crease', or 'other'
        """
        if not color:
            return 'other'
        
        if color in self.cut_colors:
            return 'cut'
        elif color in self.crease_colors:
            return 'crease'
        else:
            return 'other'
    
    def filter(self, svg_path: Path) -> Dict[str, List[Dict]]:
        """
        Parse SVG and extract semantic path data.
        
        Args:
            svg_path: Path to SVG file
        
        Returns:
            Dictionary with 'cut', 'crease', and 'other' lists containing path data
        
        Raises:
            FileNotFoundError: If SVG file doesn't exist
            ET.ParseError: If SVG is malformed
        """
        if not svg_path.exists():
            raise FileNotFoundError(f"SVG file not found: {svg_path}")
        
        self.logger.info(f"Filtering paths from {svg_path.name}...")
        
        # Parse XML
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Store paths by category
        paths = {
            'cut': [],
            'crease': [],
            'other': []
        }
        
        # Find all path elements (with and without namespace)
        path_elements = (
            root.findall('.//svg:path', self.SVG_NS) +
            root.findall('.//path')
        )
        
        for path_elem in path_elements:
            path_data = path_elem.get('d')
            if not path_data:
                continue
            
            # Skip if path is filled (we only want strokes)
            fill = path_elem.get('fill', '').lower()
            if fill and fill not in ['none', 'transparent', '']:
                continue
            
            # Get color and classify
            color = self._get_element_color(path_elem)
            category = self._classify_path(color)
            
            # Store path info
            paths[category].append({
                'data': path_data,
                'color': color,
                'id': path_elem.get('id', ''),
                'transform': path_elem.get('transform', '')
            })
        
        # Log statistics
        self.logger.info(
            f"Extracted paths: {len(paths['cut'])} cut, "
            f"{len(paths['crease'])} crease, {len(paths['other'])} other"
        )
        
        return paths

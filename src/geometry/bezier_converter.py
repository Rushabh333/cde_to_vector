"""Bézier curve converter for primitive canonicalization."""
import numpy as np
from typing import List, Tuple, Optional
import logging
from svgpathtools import parse_path, Line, QuadraticBezier, CubicBezier, Arc, Path
import math


class BezierConverter:
    """Convert SVG primitives to canonical cubic Bézier curves."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the Bézier converter.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def _line_to_cubic(self, line: Line) -> CubicBezier:
        """
        Convert a line to a cubic Bézier curve.
        
        Uses the formula:
        P0 = L0
        P1 = (2*L0 + L1) / 3
        P2 = (L0 + 2*L1) / 3
        P3 = L1
        
        Args:
            line: Line primitive
        
        Returns:
            Equivalent cubic Bézier curve
        """
        L0 = line.start
        L1 = line.end
        
        P0 = L0
        P1 = (2 * L0 + L1) / 3
        P2 = (L0 + 2 * L1) / 3
        P3 = L1
        
        return CubicBezier(P0, P1, P2, P3)
    
    def _quadratic_to_cubic(self, quad: QuadraticBezier) -> CubicBezier:
        """
        Convert a quadratic Bézier to a cubic Bézier using degree elevation.
        
        For quadratic with control points Q0, Q1, Q2:
        P0 = Q0
        P1 = Q0 + 2/3 * (Q1 - Q0)
        P2 = Q2 + 2/3 * (Q1 - Q2)
        P3 = Q2
        
        Args:
            quad: Quadratic Bézier curve
        
        Returns:
            Equivalent cubic Bézier curve
        """
        Q0 = quad.start
        Q1 = quad.control
        Q2 = quad.end
        
        P0 = Q0
        P1 = Q0 + (2/3) * (Q1 - Q0)
        P2 = Q2 + (2/3) * (Q1 - Q2)
        P3 = Q2
        
        return CubicBezier(P0, P1, P2, P3)
    
    def _arc_to_cubics(self, arc: Arc, max_angle: float = 90.0) -> List[CubicBezier]:
        """
        Convert an arc to cubic Bézier curves.
        
        Uses a split strategy: arcs are divided so that each segment
        has an angle ≤ max_angle to minimize approximation error.
        
        Args:
            arc: Arc primitive
            max_angle: Maximum angle per segment in degrees
        
        Returns:
            List of cubic Bézier curves approximating the arc
        """
        # Get the arc's total angle
        total_angle = abs(arc.theta - arc.phi)
        if total_angle > 180:
            total_angle = 360 - total_angle
        
        # Calculate number of segments needed
        num_segments = max(1, int(np.ceil(total_angle / max_angle)))
        
        cubics = []
        for i in range(num_segments):
            t_start = i / num_segments
            t_end = (i + 1) / num_segments
            
            # Sample the arc at subdivided points
            p0 = arc.point(t_start)
            p3 = arc.point(t_end)
            
            # Calculate control points using arc tangents
            # This is an approximation using derivatives
            t_mid_start = t_start + 0.001
            t_mid_end = t_end - 0.001
            
            tangent_start = (arc.point(t_mid_start) - p0) * 100
            tangent_end = (p3 - arc.point(t_mid_end)) * 100
            
            # Control points at 1/3 and 2/3 along tangent direction
            p1 = p0 + tangent_start / 3
            p2 = p3 - tangent_end / 3
            
            cubics.append(CubicBezier(p0, p1, p2, p3))
        
        return cubics
    
    def convert_path(self, path_data: str) -> List[CubicBezier]:
        """
        Convert an SVG path to a list of cubic Bézier curves.
        
        Args:
            path_data: SVG path 'd' attribute string
        
        Returns:
            List of cubic Bézier curves
        
        Raises:
            ValueError: If path data is invalid
        """
        try:
            path = parse_path(path_data)
        except Exception as e:
            raise ValueError(f"Failed to parse path data: {e}")
        
        cubics = []
        
        for segment in path:
            if isinstance(segment, CubicBezier):
                # Already a cubic Bézier
                cubics.append(segment)
            elif isinstance(segment, Line):
                # Convert line to cubic
                cubics.append(self._line_to_cubic(segment))
            elif isinstance(segment, QuadraticBezier):
                # Convert quadratic to cubic
                cubics.append(self._quadratic_to_cubic(segment))
            elif isinstance(segment, Arc):
                # Convert arc to multiple cubics
                cubics.extend(self._arc_to_cubics(segment))
            else:
                self.logger.warning(f"Unknown segment type: {type(segment)}, skipping")
        
        return cubics
    
    def extract_control_points(self, cubic: CubicBezier) -> np.ndarray:
        """
        Extract control points from a cubic Bézier curve.
        
        Args:
            cubic: Cubic Bézier curve
        
        Returns:
            Array of 8 values: [sx, sy, cx1, cy1, cx2, cy2, ex, ey]
        """
        return np.array([
            cubic.start.real, cubic.start.imag,
            cubic.control1.real, cubic.control1.imag,
            cubic.control2.real, cubic.control2.imag,
            cubic.end.real, cubic.end.imag
        ], dtype=np.float32)

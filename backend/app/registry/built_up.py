"""
SatQuery AI - Built-Up & Urban Extent Specialist
Extracts urban settlements, buildings, and impervious surfaces
using high-frequency spatial morphology and spectral response.
"""

import numpy as np
import cv2
from typing import Dict, Any, List
from app.config import DEVICE
from app.registry.base import BaseSpecialist

class BuiltUpSpecialist(BaseSpecialist):
    def __init__(self):
        super().__init__(
            name="UrbanMorphology-BuiltUp",
            version="1.0.0",
            task="BUILT_UP_DETECTION"
        )
        self.device = DEVICE

    def validate_inputs(self, images: List) -> bool:
        return len(images) >= 1 and images[0] is not None

    def execute(self, images: List, query: str, parameters: dict, metadata: List) -> Dict[str, Any]:
        image_arr = images[0]
        meta = metadata[0] if metadata else {}
        return self.analyze_built_up(image_arr, meta)

    def analyze_built_up(self, image_arr: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Extracts built-up urban structures, residential footprints,
        and high-density impervious surfaces.
        """
        h, w = image_arr.shape[:2]
        total_pixels = h * w
        
        if image_arr.ndim == 3 and image_arr.shape[2] >= 3:
            rgb_base = cv2.normalize(image_arr[:, :, :3], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            gray = cv2.cvtColor(rgb_base, cv2.COLOR_RGB2GRAY)
        else:
            g = cv2.normalize(image_arr if image_arr.ndim == 2 else image_arr[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            rgb_base = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
            gray = g

        # Urban areas have high local standard deviation and edge density
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_mag = np.sqrt(sobelx**2 + sobely**2)
        norm_edge = cv2.normalize(edge_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        # Morphological gradient to isolate compact rectangular footprints
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        gradient = cv2.morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
        
        combined = cv2.addWeighted(norm_edge, 0.6, gradient, 0.4, 0)
        _, built_mask = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Eliminate isolated noise
        clean_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        built_mask = cv2.morphologyEx(built_mask, cv2.MORPH_OPEN, clean_kernel)

        built_pixels = int(np.count_nonzero(built_mask))
        built_pct = round((built_pixels / total_pixels) * 100.0, 2)
        built_detected = built_pct > 0.5

        # Find individual structural contours / building clusters
        contours, _ = cv2.findContours(built_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        significant_clusters = [c for c in contours if cv2.contourArea(c) > 25]

        # Generate overlay in bold Crimson-Red (RGB: 230, 40, 40)
        overlay = rgb_base.copy()
        overlay[built_mask > 0] = [230, 40, 40]
        blended = cv2.addWeighted(rgb_base, 0.55, overlay, 0.45, 0)

        answer = (
            f"Built-Up & Urban Analysis: Urban settlements and built-up structures occupy {built_pct}% "
            f"of the scene ({built_pixels:,} pixels), with {len(significant_clusters)} distinct structural clusters detected."
        ) if built_detected else (
            f"Built-Up & Urban Analysis: Minimal or no urban development detected "
            f"(built-up coverage is {built_pct}%)."
        )

        return {
            "built_up_detected": built_detected,
            "built_up_percentage": built_pct,
            "built_up_area_pixels": built_pixels,
            "clusters_count": len(significant_clusters),
            "mask": built_mask,
            "overlay": blended,
            "overlay_image": blended,
            "confidence": 0.88,
            "answer": answer,
            "provenance": {
                "algorithm": "Spatial Morphology & Gradient Edge Density Filtering",
                "confidence": 0.88
            }
        }

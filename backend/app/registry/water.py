"""
SatQuery AI - Water Specialist
Implements Normalized Difference Water Index (NDWI / MNDWI)
and Optical Hydro-Segmentation adhering to BaseSpecialist.
"""

import numpy as np
import cv2
from typing import Dict, Any, List
from app.config import DEVICE
from app.registry.base import BaseSpecialist

class WaterSpecialist(BaseSpecialist):
    def __init__(self):
        super().__init__(
            name="HydroSegmentation-NDWI",
            version="1.0.0",
            task="WATER_DETECTION"
        )
        self.device = DEVICE

    def validate_inputs(self, images: List) -> bool:
        return len(images) >= 1 and images[0] is not None

    def execute(self, images: List, query: str, parameters: dict, metadata: List) -> Dict[str, Any]:
        image_arr = images[0]
        meta = metadata[0] if metadata else {}
        return self.analyze_water(image_arr, meta)

    def analyze_water(self, image_arr: np.ndarray, metadata: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Detects surface water bodies using NDWI on multi-band NIR data
        or adaptive hydro-segmentation on optical RGB.
        """
        h, w = image_arr.shape[:2]
        total_pixels = h * w
        meta = metadata or {}
        num_bands = image_arr.shape[2] if image_arr.ndim == 3 else 1

        if num_bands >= 4:
            # Band 2 = Green (index 1), Band 4 = NIR (index 3)
            # McFeeters NDWI = (Green - NIR) / (Green + NIR)
            green = image_arr[:, :, 1].astype(np.float32)
            nir = image_arr[:, :, 3].astype(np.float32)
            denom = green + nir
            denom[denom == 0] = 1e-5
            ndwi = (green - nir) / denom
            
            # Water has positive NDWI
            water_mask = (ndwi > 0.05).astype(np.uint8) * 255
            method = "McFeeters Normalized Difference Water Index (NDWI = (Green - NIR) / (Green + NIR))"
            provenance_bands = ["Band 2 (Green)", "Band 4 (NIR)"]
        else:
            # 3-band RGB imagery
            if image_arr.ndim == 3:
                r = image_arr[:, :, 0].astype(np.float32)
                g = image_arr[:, :, 1].astype(np.float32)
                b = image_arr[:, :, 2].astype(np.float32)
                # Water reflectance: High in blue/green, low in red
                blue_green = (b + g) / 2.0
                contrast = (blue_green - r) / (blue_green + r + 1e-5)
                norm_contrast = cv2.normalize(contrast, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, water_mask = cv2.threshold(norm_contrast, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                method = "Optical RGB Hydro-Spectral Contrast Filter"
                provenance_bands = ["Band 1 (Red)", "Band 2 (Green)", "Band 3 (Blue)"]
            else:
                norm_gray = cv2.normalize(image_arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                _, water_mask = cv2.threshold(norm_gray, 50, 255, cv2.THRESH_BINARY_INV)
                method = "Single-band Low Backscatter / Reflectance Thresholding"
                provenance_bands = ["Single Band"]

        # Clean noise with morphological opening
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        water_mask = cv2.morphologyEx(water_mask, cv2.MORPH_OPEN, kernel)

        water_pixels = int(np.count_nonzero(water_mask))
        water_pct = round((water_pixels / total_pixels) * 100.0, 2)
        water_detected = water_pct > 0.1

        # Generate colored overlay
        if image_arr.ndim == 3 and image_arr.shape[2] >= 3:
            rgb_base = cv2.normalize(image_arr[:, :, :3], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        else:
            g = cv2.normalize(image_arr if image_arr.ndim == 2 else image_arr[:, :, 0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            rgb_base = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)

        overlay = rgb_base.copy()
        # Water in vibrant Cyan-Blue (RGB: 0, 180, 255)
        overlay[water_mask > 0] = [0, 180, 255]
        blended = cv2.addWeighted(rgb_base, 0.55, overlay, 0.45, 0)

        answer = (
            f"Surface Water Analysis: {water_pct}% of the surveyed area is covered by surface water bodies "
            f"({water_pixels:,} pixels out of {total_pixels:,}). Method: {method}."
        ) if water_detected else (
            f"Surface Water Analysis: No significant open water bodies were detected in this scene "
            f"(measured coverage < 0.1%). Method: {method}."
        )

        return {
            "water_detected": water_detected,
            "water_percentage": water_pct,
            "water_area_pixels": water_pixels,
            "method": method,
            "mask": water_mask,
            "overlay": blended,
            "overlay_image": blended,
            "confidence": 0.94 if num_bands >= 4 else 0.85,
            "answer": answer,
            "provenance": {
                "algorithm": method,
                "bands_used": provenance_bands,
                "confidence": 0.94 if num_bands >= 4 else 0.85
            }
        }

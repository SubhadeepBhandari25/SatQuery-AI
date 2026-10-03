import numpy as np
import cv2
from app.registry.base import BaseSpecialist
from app.evidence.engine import mask_to_colored_overlay

class SpectralVegetationSpecialist(BaseSpecialist):
    """
    Specialist for vegetation intelligence and spectral indices.
    Calculates genuine NDVI when NIR and Red bands are present.
    Generates false-color NDVI heatmap and classifies vegetation structure.
    Strictly refuses to compute fake NDVI on RGB-only imagery.
    """
    def __init__(self):
        super().__init__(
            name="Spectral-Vegetation-NDVI-Engine",
            version="2.0.0",
            task="VEGETATION_ANALYSIS"
        )

    def validate_inputs(self, images: list) -> bool:
        return len(images) >= 1 and images[0] is not None

    def execute(self, images: list, query: str, parameters: dict, metadata: list) -> dict:
        img = images[0]
        meta = metadata[0] if metadata else {}
        h, w = img.shape[:2]
        bands = img.shape[2] if img.ndim == 3 else 1
        
        has_nir = (bands >= 4)
        
        if has_nir:
            # Band 1: Red, Band 4: NIR (Sentinel-2 / Landsat / Cartosat 4-band standard)
            red = img[:, :, 0].astype(np.float32)
            nir = img[:, :, 3].astype(np.float32)
            
            denom = nir + red
            denom[denom == 0] = 1e-5
            ndvi = (nir - red) / denom
            # Clip to valid mathematical range [-1.0, 1.0]
            ndvi = np.clip(ndvi, -1.0, 1.0)
            
            # Mask out water or extreme negative values for vegetation statistics
            veg_mask = ndvi > 0.12
            dense_mask = ndvi >= 0.55
            cropland_mask = (ndvi >= 0.30) & (ndvi < 0.55)
            sparse_mask = (ndvi >= 0.12) & (ndvi < 0.30)
            
            total_px = h * w
            veg_px = np.sum(veg_mask)
            veg_pct = float(veg_px / total_px * 100.0)
            dense_pct = float(np.sum(dense_mask) / total_px * 100.0)
            cropland_pct = float(np.sum(cropland_mask) / total_px * 100.0)
            sparse_pct = float(np.sum(sparse_mask) / total_px * 100.0)
            
            mean_ndvi = float(np.mean(ndvi[veg_mask])) if veg_px > 0 else 0.0
            
            # Generate False-Color NDVI Map (RdYlGn palette via OpenCV)
            # Map [-0.2, 0.8] to [0, 255]
            ndvi_scaled = np.clip((ndvi + 0.2) / 1.0 * 255.0, 0, 255).astype(np.uint8)
            ndvi_colormap = cv2.applyColorMap(ndvi_scaled, cv2.COLORMAP_SUMMER)
            # Invert so high NDVI is deep emerald green
            ndvi_vis = ndvi_colormap.copy()
            ndvi_vis[:, :, 0] = 255 - ndvi_colormap[:, :, 2] # Blue
            ndvi_vis[:, :, 1] = ndvi_colormap[:, :, 1]       # Green
            ndvi_vis[:, :, 2] = 255 - ndvi_colormap[:, :, 0] # Red

            if dense_pct > 25.0:
                condition = "Dense vegetative canopy / Healthy active biomass"
            elif cropland_pct > 20.0:
                condition = "Active agricultural cropland / Moderate canopy cover"
            elif veg_pct > 10.0:
                condition = "Sparse grassland / Shrubland vegetation"
            else:
                condition = "Minimal vegetative cover / Predominantly non-vegetated terrain"

            answer = (
                f"Spectral vegetation analysis successfully calculated Normalized Difference Vegetation Index (NDVI) "
                f"using Red (Band 1, 665nm) and Near-Infrared (Band 4, 842nm). "
                f"Vegetation occupies {veg_pct:.1f}% of the scene (Mean NDVI: {mean_ndvi:.2f}). "
                f"Breakdown: Dense forest/canopy: {dense_pct:.1f}%, Cropland/Pasture: {cropland_pct:.1f}%, "
                f"Sparse greenery: {sparse_pct:.1f}%. Overall vegetative status: {condition}."
            )

            return {
                "answer": answer,
                "confidence": 0.94,
                "ndvi_available": True,
                "mean_ndvi": round(mean_ndvi, 3),
                "max_ndvi": round(float(np.max(ndvi)), 3),
                "min_ndvi": round(float(np.min(ndvi)), 3),
                "vegetation_coverage_pct": round(veg_pct, 1),
                "dense_canopy_pct": round(dense_pct, 1),
                "cropland_pct": round(cropland_pct, 1),
                "sparse_veg_pct": round(sparse_pct, 1),
                "condition": condition,
                "vegetation_mask": veg_mask.astype(np.uint8) * 255,
                "ndvi_map": ndvi_vis,
                "provenance": {
                    "method": "Normalized Difference Vegetation Index (NDVI)",
                    "formula": "(NIR - Red) / (NIR + Red)",
                    "source": "Image Multispectral Analysis (Band 1 + Band 4)",
                    "type": "image_derived"
                }
            }
        else:
            # 3-band RGB fallback: Excess Green Index (ExG)
            rgb = img[:, :, :3].astype(np.float32) / 255.0 if img.ndim == 3 else np.stack([img]*3, axis=-1).astype(np.float32) / 255.0
            r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
            exg = 2.0 * g - r - b
            veg_mask = exg > 0.08
            veg_pct = float(np.sum(veg_mask) / (h * w) * 100.0)
            
            rgb_uint8 = (rgb * 255).astype(np.uint8)
            overlay = mask_to_colored_overlay(rgb_uint8, veg_mask.astype(np.uint8) * 255, color_rgb=(40, 200, 60), alpha=0.45)
            
            answer = (
                f"Vegetation detected via optical Excess Green Index (ExG), covering approximately {veg_pct:.1f}% "
                f"of the analyzed scene. Note: Standard NDVI is unavailable because the uploaded image is 3-band RGB "
                f"and does not contain a calibrated Near-Infrared (NIR) band. Synthetic NDVI was strictly not fabricated."
            )

            return {
                "answer": answer,
                "confidence": 0.85,
                "ndvi_available": False,
                "mean_ndvi": None,
                "reason_ndvi_unavailable": "NDVI requires Near-Infrared (NIR) band and cannot be calculated from RGB-only imagery.",
                "vegetation_coverage_pct": round(veg_pct, 1),
                "condition": "Optical greenness detected (Visible spectrum)",
                "vegetation_mask": veg_mask.astype(np.uint8) * 255,
                "ndvi_map": overlay,
                "provenance": {
                    "method": "Excess Green Index (ExG = 2G - R - B)",
                    "source": "Optical RGB Visual Spectrum Analysis",
                    "type": "image_derived"
                }
            }

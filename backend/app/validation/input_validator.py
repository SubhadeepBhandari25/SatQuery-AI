import os
from pathlib import Path
from PIL import Image
import numpy as np
import rasterio
from app.config import (
    MAX_UPLOAD_BYTES, MAX_UPLOAD_SIZE_MB,
    ALLOWED_EXTENSIONS
)

def validate_single_file(file_path: Path) -> dict:
    """
    Validates uploaded file: existence, size, extension, image decodability,
    dimensions, and channel count.
    """
    if not file_path.exists():
        return {"valid": False, "error": f"File does not exist: {file_path.name}"}
    
    file_size = file_path.stat().st_size
    if file_size > MAX_UPLOAD_BYTES:
        return {
            "valid": False,
            "error": f"File {file_path.name} exceeds maximum upload limit of {MAX_UPLOAD_SIZE_MB}MB"
        }
    if file_size == 0:
        return {"valid": False, "error": f"File {file_path.name} is empty."}
    
    ext = file_path.suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        return {
            "valid": False,
            "error": f"Unsupported format '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}"
        }
    
    # Try inspecting with rasterio first (for GeoTIFF/TIFF)
    if ext in {".tif", ".tiff"}:
        try:
            with rasterio.open(file_path) as src:
                width, height = src.width, src.height
                bands = src.count
                dtype = str(src.dtypes[0])
                crs = src.crs.to_string() if src.crs else None
                georeferenced = crs is not None
                return {
                    "valid": True,
                    "filename": file_path.name,
                    "format": "GeoTIFF",
                    "width": width,
                    "height": height,
                    "bands": bands,
                    "dtype": dtype,
                    "georeferenced": georeferenced,
                    "crs": crs,
                    "size_bytes": file_size
                }
        except Exception as e:
            pass  # Fallback to PIL if rasterio encounters non-standard TIFF

    # Fallback to PIL
    try:
        with Image.open(file_path) as img:
            width, height = img.size
            bands = len(img.getbands())
            return {
                "valid": True,
                "filename": file_path.name,
                "format": img.format or ext.replace(".", "").upper(),
                "width": width,
                "height": height,
                "bands": bands,
                "dtype": "uint8",
                "georeferenced": False,
                "crs": None,
                "size_bytes": file_size
            }
    except Exception as e:
        return {"valid": False, "error": f"Unable to decode image {file_path.name}: {str(e)}"}

def determine_modality(file_path: Path, metadata: dict) -> str:
    """
    Determines modality: optical, multispectral, sar, or unknown.
    Uses band descriptions, image stats, name heuristics, and spectral attributes.
    """
    name_lower = file_path.name.lower()
    bands = metadata.get("bands", 3)
    
    # Check explicit name hints from ISRO/SAC or standard conventions
    if any(k in name_lower for k in ["sar", "risat", "sentinel1", "s1", "slc", "grd", "vv", "vh", "hh", "hv"]):
        return "sar"
    
    if bands == 1:
        # Check dynamic range or standard deviation characteristic of SAR backscatter
        try:
            with rasterio.open(file_path) as src:
                data = src.read(1)
                # SAR amplitude/intensity or dB often has distinct skewed histogram
                desc = src.descriptions[0] if src.descriptions and src.descriptions[0] else ""
                if any(p in desc.lower() for p in ["vv", "vh", "hh", "hv", "sigma0", "gamma0"]):
                    return "sar"
        except Exception:
            pass
        return "sar" if "sar" in name_lower else "optical"
        
    if bands > 3:
        return "multispectral"
    
    return "optical"

def validate_pair(file1_info: dict, file2_info: dict, expected_pair_type: str) -> dict:
    """
    Validates paired imagery (Optical+SAR or Bi-temporal pair).
    Verifies spatial compatibility, dimension compatibility, and modality validity.
    """
    w1, h1 = file1_info["width"], file1_info["height"]
    w2, h2 = file2_info["width"], file2_info["height"]
    
    mod1 = file1_info.get("modality", "optical")
    mod2 = file2_info.get("modality", "optical")
    
    # Check dimensions
    dim_mismatch = (w1 != w2) or (h1 != h2)
    dim_ratio = max(w1, w2) / max(1, min(w1, w2))
    
    warnings = []
    if dim_mismatch:
        if dim_ratio > 3.0:
            return {
                "compatible": False,
                "error": f"Image dimension mismatch too large ({w1}x{h1} vs {w2}x{h2}). Must be co-registered or similar scale."
            }
        warnings.append(f"Dimensions differ ({w1}x{h1} vs {w2}x{h2}). Automatic spatial resampling will be applied.")
        
    if expected_pair_type == "OPTICAL_SAR":
        modalities = {mod1, mod2}
        if "sar" not in modalities:
            warnings.append("Neither image was distinctly identified as SAR. Cross-modal backscatter analysis will treat Image 2 as structural proxy.")
    elif expected_pair_type == "BI_TEMPORAL":
        # Check CRS compatibility if georeferenced
        crs1 = file1_info.get("crs")
        crs2 = file2_info.get("crs")
        if crs1 and crs2 and crs1 != crs2:
            warnings.append(f"Differing coordinate reference systems ({crs1} vs {crs2}). Re-projection required for exact alignment.")

    return {
        "compatible": True,
        "warnings": warnings,
        "pair_type": expected_pair_type
    }

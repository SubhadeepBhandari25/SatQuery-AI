import rasterio.transform
from pyproj import Transformer

def pixel_to_geographic(transform_list: list, crs_str: str, px: float, py: float) -> tuple[float, float] | None:
    """
    Transforms pixel coordinate (px, py) to WGS84 geographic (latitude, longitude).
    Uses rasterio affine transform and pyproj reprojection.
    Never invents coordinates. Returns None if georeferencing is missing.
    """
    if not transform_list or not crs_str:
        return None
    try:
        t = rasterio.Affine(*transform_list)
        # Pixel to projected coordinate (x, y)
        proj_x, proj_y = rasterio.transform.xy(t, py, px)
        
        # Transform projected to EPSG:4326 (lat, lon)
        transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(proj_x, proj_y)
        return float(lat), float(lon)
    except Exception:
        return None

def get_wgs84_bounds(bounds_dict: dict, crs_str: str) -> dict | None:
    """
    Reprojects bounding box coordinates to WGS84 [min_lat, min_lon, max_lat, max_lon].
    """
    if not bounds_dict or not crs_str:
        return None
    try:
        transformer = Transformer.from_crs(crs_str, "EPSG:4326", always_xy=True)
        min_lon, min_lat = transformer.transform(bounds_dict["left"], bounds_dict["bottom"])
        max_lon, max_lat = transformer.transform(bounds_dict["right"], bounds_dict["top"])
        return {
            "min_lat": float(min_lat),
            "min_lon": float(min_lon),
            "max_lat": float(max_lat),
            "max_lon": float(max_lon),
            "center_lat": float((min_lat + max_lat) / 2.0),
            "center_lon": float((min_lon + max_lon) / 2.0)
        }
    except Exception:
        return None

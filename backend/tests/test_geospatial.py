import pytest
from pathlib import Path
from app.geospatial.metadata_extractor import extract_metadata
from app.geospatial.coordinate_transform import pixel_to_geographic, get_wgs84_bounds
from sample_data.generate_sample_data import DATA_DIR

def test_extract_geotiff_metadata():
    geo_path = DATA_DIR / "optical_sample_delhi.tif"
    meta = extract_metadata(geo_path)
    assert meta["georeferenced"] is True
    assert meta["crs"] is not None
    assert meta["bounds"] is not None
    assert meta["transform"] is not None
    assert len(meta["transform"]) == 6

def test_pixel_to_geographic_transformation():
    geo_path = DATA_DIR / "optical_sample_delhi.tif"
    meta = extract_metadata(geo_path)
    # Test pixel center conversion
    lat_lon = pixel_to_geographic(meta["transform"], meta["crs"], 256, 256)
    assert lat_lon is not None
    lat, lon = lat_lon
    # UTM 43N (Delhi region) latitude ~ 28.5, longitude ~ 77.2
    assert 20.0 < lat < 35.0
    assert 70.0 < lon < 85.0

def test_no_fake_coordinates_for_png():
    png_path = DATA_DIR / "sample_non_geo.png"
    meta = extract_metadata(png_path)
    assert meta["georeferenced"] is False
    assert meta["crs"] is None
    assert "unavailable" in meta["message"].lower()
    
    # Coordinate transform must return None
    lat_lon = pixel_to_geographic(meta["transform"], meta["crs"], 100, 100)
    assert lat_lon is None

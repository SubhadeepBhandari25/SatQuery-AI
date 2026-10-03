import pytest
from pathlib import Path
from app.validation.input_validator import validate_single_file, determine_modality, validate_pair
from sample_data.generate_sample_data import DATA_DIR, generate_all_samples

@pytest.fixture(scope="session", autouse=True)
def setup_samples():
    generate_all_samples()

def test_validate_geotiff():
    geo_path = DATA_DIR / "optical_sample_delhi.tif"
    res = validate_single_file(geo_path)
    assert res["valid"] is True
    assert res["format"] == "GeoTIFF"
    assert res["width"] == 512
    assert res["height"] == 512
    assert res["bands"] == 4
    assert res["georeferenced"] is True
    assert "32643" in res["crs"]

def test_validate_non_geo_png():
    png_path = DATA_DIR / "sample_non_geo.png"
    res = validate_single_file(png_path)
    assert res["valid"] is True
    assert res["georeferenced"] is False
    assert res["crs"] is None

def test_determine_modality():
    opt_path = DATA_DIR / "optical_sample_delhi.tif"
    sar_path = DATA_DIR / "sar_sample_delhi.tif"
    
    assert determine_modality(opt_path, {"bands": 4}) == "multispectral"
    assert determine_modality(sar_path, {"bands": 1}) == "sar"

def test_validate_pair_compatibility():
    t1 = {"width": 512, "height": 512, "modality": "optical"}
    t2 = {"width": 512, "height": 512, "modality": "optical"}
    pair_res = validate_pair(t1, t2, "BI_TEMPORAL")
    assert pair_res["compatible"] is True
    assert len(pair_res["warnings"]) == 0

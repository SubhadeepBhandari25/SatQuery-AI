import rasterio
import numpy as np
from pathlib import Path
from app.registry.grounding import GroundingSpecialist
from app.registry.change_detection import ChangeDetectionSpecialist
from app.registry.optical_sar import OpticalSARFusionSpecialist
from app.registry.vqa import RSVQASpecialist
from sample_data.generate_sample_data import DATA_DIR

def load_tif(filename):
    with rasterio.open(DATA_DIR / filename) as src:
        d = src.read()
        return d[0] if d.shape[0] == 1 else np.transpose(d, (1, 2, 0))

def test_grounding_specialist_water():
    img = load_tif("optical_sample_delhi.tif")
    specialist = GroundingSpecialist()
    res = specialist.execute([img], "Highlight water body", {"target": "water"}, [{}])
    
    assert res["mask"] is not None
    assert res["coverage_percent"] > 0.5
    assert len(res["regions"]) > 0
    assert res["confidence"] is not None
    assert "water body" in res["answer"].lower()

def test_change_detection_specialist():
    t1 = load_tif("bitemporal_t1.tif")
    t2 = load_tif("bitemporal_t2.tif")
    specialist = ChangeDetectionSpecialist()
    res = specialist.execute([t1, t2], "What changed?", {}, [{}, {}])
    
    assert res["change_percentage"] > 0.1
    assert res["changed_pixels"] > 0
    assert len(res["regions"]) > 0
    assert res["composite_image"] is not None

def test_change_vqa_specialist():
    t1 = load_tif("bitemporal_t1.tif")
    t2 = load_tif("bitemporal_t2.tif")
    specialist = ChangeDetectionSpecialist()
    res = specialist.execute([t1, t2], "Has the built-up area increased?", {}, [{}, {}])
    assert "built-up" in res["answer"].lower()

def test_optical_sar_fusion_specialist():
    opt = load_tif("optical_sample_delhi.tif")
    sar = load_tif("sar_sample_delhi.tif")
    specialist = OpticalSARFusionSpecialist()
    res = specialist.execute([opt, sar], "Perform optical and SAR fusion", {}, [{}, {}])
    
    assert res["built_up_percentage"] > 0
    assert res["water_percentage"] > 0
    assert res["fused_composite"] is not None
    assert "cross-modal" in res["answer"].lower()

def test_vqa_specialist():
    img = load_tif("optical_sample_delhi.tif")
    specialist = RSVQASpecialist()
    res = specialist.execute([img], "What is visible in this satellite image?", {}, [{}])
    
    assert len(res["answer"]) > 20
    assert res["confidence"] is not None
    assert len(res["bigearthnet_categories"]) > 0

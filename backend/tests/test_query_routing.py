from app.agent.query_analyzer import analyze_query

def test_vqa_routing():
    res = analyze_query("What is visible in this satellite image?", num_images=1)
    assert res["task"] == "SINGLE_IMAGE_VQA"

def test_grounding_water():
    res = analyze_query("Highlight the water body", num_images=1)
    assert res["task"] == "TEXT_GUIDED_GROUNDING"
    assert res["target_class"] == "water"

def test_grounding_buildings():
    res = analyze_query("Where are the buildings located in this area?", num_images=1)
    assert res["task"] == "TEXT_GUIDED_GROUNDING"
    assert res["target_class"] == "built-up"

def test_bitemporal_change():
    res = analyze_query("What changed between these two images?", num_images=2, modalities=["optical", "optical"])
    assert res["task"] == "BI_TEMPORAL_CHANGE"

def test_change_vqa():
    res = analyze_query("Has the built-up area increased?", num_images=2, modalities=["optical", "optical"])
    assert res["task"] == "CHANGE_VQA"

def test_optical_sar_routing():
    res = analyze_query("Compare optical and SAR images", num_images=2, modalities=["optical", "sar"])
    assert res["task"] == "OPTICAL_SAR_ANALYSIS"

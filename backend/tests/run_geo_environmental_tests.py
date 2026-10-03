import sys
import json
from pathlib import Path

backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.storage.file_store import generate_session_id, save_uploaded_file, get_result, get_report_path
from app.agent.controller import controller
from sample_data.generate_sample_data import DATA_DIR, generate_all_samples

def run_tests():
    print("====================================================================")
    print(" SATQUERY AI — GEO-SPATIAL + ENVIRONMENTAL INTELLIGENCE VERIFICATION")
    print("====================================================================\n")
    generate_all_samples()
    
    geo_file = DATA_DIR / "optical_sample_delhi.tif"
    non_geo_file = DATA_DIR / "sample_non_geo.png"
    
    results = {}

    # TEST A: Scene Intelligence Report
    print(">>> RUNNING TEST A: SCENE INTELLIGENCE REPORT")
    s_a = generate_session_id()
    save_uploaded_file(s_a, geo_file.name, geo_file.read_bytes())
    res_a = controller.process_analysis(
        analysis_id=s_a,
        query="Give me information about this image.",
        filenames=[geo_file.name]
    )
    assert res_a["task"] == "SCENE_INTELLIGENCE_REPORT", f"Unexpected task: {res_a['task']}"
    assert len(res_a["answer"]) > 50, "Answer too short"
    assert res_a["land_cover"]["classes"]["water_pct"] > 0
    assert res_a["land_cover"]["classes"]["total_vegetation_pct"] > 0
    assert res_a["summary"]["headline"] is not None
    assert len(res_a["visual_evidence"]) >= 2
    print(f"  [PASS] Task: {res_a['task']}")
    print(f"  [PASS] Briefing: {res_a['answer'][:110]}...")
    print(f"  [PASS] Land Cover: Water={res_a['land_cover']['classes']['water_pct']}%, Veg={res_a['land_cover']['classes']['total_vegetation_pct']}%")
    results["TEST_A_SCENE_REPORT"] = "PASS"

    # TEST B: Geographic Location & Reverse Geocoding
    print("\n>>> RUNNING TEST B: WHERE IS THIS IMAGE? (GEOLOCATION)")
    s_b = generate_session_id()
    save_uploaded_file(s_b, geo_file.name, geo_file.read_bytes())
    res_b = controller.process_analysis(
        analysis_id=s_b,
        query="Where is this image?",
        filenames=[geo_file.name]
    )
    assert res_b["task"] == "LOCATION_QUERY"
    loc = res_b["location"]
    assert loc["latitude"] is not None and loc["longitude"] is not None
    assert "India" in (loc["country"] or "") or "Asia" in (loc["continent"] or "")
    assert loc["provenance"]["source"] is not None
    print(f"  [PASS] Coordinates: {loc['latitude']}° N, {loc['longitude']}° E (WGS84)")
    print(f"  [PASS] Resolved Location: {loc.get('nearest_city') or loc.get('state')}, {loc.get('country')} ({loc.get('continent')})")
    print(f"  [PASS] Source Provenance: {loc['provenance']['source']}")
    results["TEST_B_LOCATION"] = "PASS"

    # TEST C: Vegetation Intelligence & True NDVI
    print("\n>>> RUNNING TEST C: VEGETATION & NDVI ANALYSIS")
    s_c = generate_session_id()
    save_uploaded_file(s_c, geo_file.name, geo_file.read_bytes())
    res_c = controller.process_analysis(
        analysis_id=s_c,
        query="What vegetation is present?",
        filenames=[geo_file.name]
    )
    assert res_c["task"] == "VEGETATION_ANALYSIS"
    veg = res_c["vegetation"]
    assert veg["ndvi_available"] is True, "4-band GeoTIFF should support NDVI"
    assert veg["mean_ndvi"] is not None
    assert veg["coverage_pct"] > 0
    assert veg["provenance"]["formula"] == "(NIR - Red) / (NIR + Red)"
    print(f"  [PASS] NDVI Available: {veg['ndvi_available']} (Mean NDVI: {veg['mean_ndvi']})")
    print(f"  [PASS] Vegetative Condition: {veg['condition']}")
    print(f"  [PASS] Provenance Formula: {veg['provenance']['formula']}")
    results["TEST_C_VEGETATION_NDVI"] = "PASS"

    # TEST D: Temperature Intelligence
    print("\n>>> RUNNING TEST D: TEMPERATURE INTELLIGENCE")
    s_d = generate_session_id()
    save_uploaded_file(s_d, geo_file.name, geo_file.read_bytes())
    res_d = controller.process_analysis(
        analysis_id=s_d,
        query="What is the temperature here?",
        filenames=[geo_file.name]
    )
    assert res_d["task"] == "TEMPERATURE_QUERY"
    temp = res_d["environment"]["temperature"]
    assert temp["unit"] == "°C"
    assert temp["status"] == "available"
    assert temp["value"] is not None
    assert "Open-Meteo" in temp["source"]
    print(f"  [PASS] Temperature: {temp['value']} {temp['unit']}")
    print(f"  [PASS] Source: {temp['source']} (Type: {temp['type']})")
    print(f"  [PASS] Answer: {res_d['answer']}")
    results["TEST_D_TEMPERATURE"] = "PASS"

    # TEST E: Moisture Intelligence
    print("\n>>> RUNNING TEST E: MOISTURE INTELLIGENCE")
    s_e = generate_session_id()
    save_uploaded_file(s_e, geo_file.name, geo_file.read_bytes())
    res_e = controller.process_analysis(
        analysis_id=s_e,
        query="What is the moisture level?",
        filenames=[geo_file.name]
    )
    assert res_e["task"] == "MOISTURE_QUERY"
    moist = res_e["environment"]["soil_moisture"]
    assert moist["unit"] == "m³/m³"
    assert moist["status"] == "available"
    assert moist["value"] is not None
    print(f"  [PASS] Soil Moisture: {moist['value']} {moist['unit']}")
    print(f"  [PASS] Source: {moist['source']} ({moist.get('spatial_resolution')})")
    print(f"  [PASS] Answer: {res_e['answer']}")
    results["TEST_E_MOISTURE"] = "PASS"

    # TEST F: Water Grounding
    print("\n>>> RUNNING TEST F: HIGHLIGHT THE WATER BODY (GROUNDING)")
    s_f = generate_session_id()
    save_uploaded_file(s_f, geo_file.name, geo_file.read_bytes())
    res_f = controller.process_analysis(
        analysis_id=s_f,
        query="Highlight the water body.",
        filenames=[geo_file.name]
    )
    assert res_f["task"] == "TEXT_GUIDED_GROUNDING"
    assert len(res_f["visual_evidence"]) > 0
    assert "water" in res_f["answer"].lower()
    print(f"  [PASS] Task: {res_f['task']}")
    print(f"  [PASS] Visual Overlays: {len(res_f['visual_evidence'])} item(s)")
    print(f"  [PASS] Answer: {res_f['answer'][:110]}...")
    results["TEST_F_GROUNDING"] = "PASS"

    # TEST G: Complete Report & Provenance Model
    print("\n>>> RUNNING TEST G: COMPLETE COMBINED REPORT")
    s_g = generate_session_id()
    save_uploaded_file(s_g, geo_file.name, geo_file.read_bytes())
    res_g = controller.process_analysis(
        analysis_id=s_g,
        query="Give me a complete report.",
        filenames=[geo_file.name]
    )
    required_sections = ["summary", "location", "metadata", "land_cover", "vegetation", "water", "built_environment", "terrain", "environment", "visual_evidence", "map", "execution_trace"]
    for sec in required_sections:
        assert sec in res_g, f"Missing required section: {sec}"
    # Verify report HTML persistence
    rep_path = get_report_path(s_g, "report.html")
    assert rep_path is not None and rep_path.exists()
    assert len(rep_path.read_text(encoding="utf-8")) > 1000
    print(f"  [PASS] All 12 Structured Sections Present in Result Model")
    print(f"  [PASS] Downloadable Standalone Report Compiled ({rep_path.stat().st_size} bytes)")
    results["TEST_G_COMPLETE_REPORT"] = "PASS"

    # TEST NON-GEO: Zero Fake Coordinates & Zero Fake Weather
    print("\n>>> RUNNING TEST NON-GEO: STANDARD PNG FALLBACK (NO FAKE DATA)")
    s_ng = generate_session_id()
    save_uploaded_file(s_ng, non_geo_file.name, non_geo_file.read_bytes())
    res_ng = controller.process_analysis(
        analysis_id=s_ng,
        query="Where is this image?",
        filenames=[non_geo_file.name]
    )
    assert res_ng["location"]["latitude"] is None
    assert res_ng["location"]["status"] == "unavailable"
    assert res_ng["environment"]["temperature"]["status"] == "unavailable"
    assert res_ng["environment"]["soil_moisture"]["status"] == "unavailable"
    assert "unavailable" in res_ng["answer"].lower()
    print(f"  [PASS] Coordinates strictly returned as None (Zero Fake Coordinates)")
    print(f"  [PASS] Temperature & Moisture strictly marked unavailable (Zero Fake Weather)")
    print(f"  [PASS] Answer: {res_ng['answer']}")
    results["TEST_NON_GEO_INTEGRITY"] = "PASS"

    print("\n====================================================================")
    print("                   FINAL VERIFICATION RESULTS                        ")
    print("====================================================================")
    for t_name, stat in results.items():
        print(f"  {t_name.ljust(28)}: {stat}")
    print("====================================================================\n")

if __name__ == "__main__":
    run_tests()

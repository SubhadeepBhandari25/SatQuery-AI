import sys
import json
from pathlib import Path

# Add backend directory to sys.path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.storage.file_store import generate_session_id, save_uploaded_file, get_result, get_report_path
from app.agent.controller import controller
from app.agent.query_analyzer import analyze_query
from app.geospatial.metadata_extractor import extract_metadata
from app.geospatial.coordinate_transform import pixel_to_geographic
from sample_data.generate_sample_data import DATA_DIR, generate_all_samples

def run_all_demo_tests():
    print("================================================================")
    print("      SATQUERY AI — SIH26167 MANDATORY DEMO VERIFICATION        ")
    print("================================================================\n")
    generate_all_samples()
    
    results = {}

    # TEST 1: SINGLE IMAGE VQA
    print(">>> RUNNING TEST 1: SINGLE IMAGE VQA")
    s1_id = generate_session_id()
    opt_file = DATA_DIR / "optical_sample_delhi.tif"
    save_uploaded_file(s1_id, opt_file.name, opt_file.read_bytes())
    t1_res = controller.process_analysis(
        analysis_id=s1_id,
        query="What is visible in this image?",
        filenames=[opt_file.name]
    )
    assert t1_res["task"] in ["SINGLE_IMAGE_VQA", "SCENE_INTELLIGENCE_REPORT"], f"Expected SINGLE_IMAGE_VQA or SCENE_INTELLIGENCE_REPORT, got {t1_res['task']}"
    assert len(t1_res["answer"]) > 20, "Answer is too short or empty"
    assert t1_res["confidence"] is not None
    print(f"  [PASS] Task: {t1_res['task']}")
    print(f"  [PASS] Answer: {t1_res['answer'][:120]}...")
    results["TEST_1_VQA"] = "PASS"

    # TEST 2: WATER GROUNDING
    print("\n>>> RUNNING TEST 2: WATER GROUNDING")
    s2_id = generate_session_id()
    save_uploaded_file(s2_id, opt_file.name, opt_file.read_bytes())
    t2_res = controller.process_analysis(
        analysis_id=s2_id,
        query="Highlight the water body.",
        filenames=[opt_file.name]
    )
    assert t2_res["task"] == "TEXT_GUIDED_GROUNDING", f"Expected TEXT_GUIDED_GROUNDING, got {t2_res['task']}"
    assert len(t2_res["visual_evidence"]) > 0, "Missing visual evidence overlays"
    assert "water" in t2_res["answer"].lower()
    print(f"  [PASS] Task: {t2_res['task']}")
    print(f"  [PASS] Evidence generated: {len(t2_res['visual_evidence'])} overlay(s)")
    print(f"  [PASS] Answer: {t2_res['answer'][:120]}...")
    results["TEST_2_GROUNDING"] = "PASS"

    # TEST 3: BI-TEMPORAL CHANGE
    print("\n>>> RUNNING TEST 3: BI-TEMPORAL CHANGE")
    s3_id = generate_session_id()
    t1_file = DATA_DIR / "bitemporal_t1.tif"
    t2_file = DATA_DIR / "bitemporal_t2.tif"
    save_uploaded_file(s3_id, t1_file.name, t1_file.read_bytes())
    save_uploaded_file(s3_id, t2_file.name, t2_file.read_bytes())
    t3_res = controller.process_analysis(
        analysis_id=s3_id,
        query="What changed between these images?",
        filenames=[t1_file.name, t2_file.name]
    )
    assert t3_res["task"] == "BI_TEMPORAL_CHANGE", f"Expected BI_TEMPORAL_CHANGE, got {t3_res['task']}"
    assert len(t3_res["visual_evidence"]) > 0
    print(f"  [PASS] Task: {t3_res['task']}")
    print(f"  [PASS] Answer: {t3_res['answer'][:120]}...")
    results["TEST_3_CHANGE"] = "PASS"

    # TEST 4: CHANGE VQA
    print("\n>>> RUNNING TEST 4: CHANGE VQA")
    s4_id = generate_session_id()
    save_uploaded_file(s4_id, t1_file.name, t1_file.read_bytes())
    save_uploaded_file(s4_id, t2_file.name, t2_file.read_bytes())
    t4_res = controller.process_analysis(
        analysis_id=s4_id,
        query="Has the built-up area increased?",
        filenames=[t1_file.name, t2_file.name]
    )
    assert t4_res["task"] == "CHANGE_VQA", f"Expected CHANGE_VQA, got {t4_res['task']}"
    assert "built-up" in t4_res["answer"].lower()
    print(f"  [PASS] Task: {t4_res['task']}")
    print(f"  [PASS] Answer: {t4_res['answer'][:120]}...")
    results["TEST_4_CHANGE_VQA"] = "PASS"

    # TEST 5: OPTICAL + SAR CROSS-MODAL FUSION
    print("\n>>> RUNNING TEST 5: OPTICAL + SAR FUSION")
    s5_id = generate_session_id()
    sar_file = DATA_DIR / "sar_sample_delhi.tif"
    save_uploaded_file(s5_id, opt_file.name, opt_file.read_bytes())
    save_uploaded_file(s5_id, sar_file.name, sar_file.read_bytes())
    t5_res = controller.process_analysis(
        analysis_id=s5_id,
        query="Identify built-up areas and water bodies using both images.",
        filenames=[opt_file.name, sar_file.name]
    )
    assert t5_res["task"] == "OPTICAL_SAR_ANALYSIS", f"Expected OPTICAL_SAR_ANALYSIS, got {t5_res['task']}"
    assert len(t5_res["visual_evidence"]) > 0
    print(f"  [PASS] Task: {t5_res['task']}")
    print(f"  [PASS] Answer: {t5_res['answer'][:120]}...")
    results["TEST_5_OPTICAL_SAR"] = "PASS"

    # TEST 6: AGENT QUERY ROUTING
    print("\n>>> RUNNING TEST 6: AGENT ROUTING")
    q_tests = [
        ("What is visible?", 1, ["optical"], "SINGLE_IMAGE_VQA"),
        ("Highlight the water body", 1, ["optical"], "TEXT_GUIDED_GROUNDING"),
        ("Where are the buildings?", 1, ["optical"], "TEXT_GUIDED_GROUNDING"),
        ("What changed?", 2, ["optical", "optical"], "BI_TEMPORAL_CHANGE"),
        ("Has the built-up area increased?", 2, ["optical", "optical"], "CHANGE_VQA"),
        ("Compare optical and SAR", 2, ["optical", "sar"], "OPTICAL_SAR_ANALYSIS"),
    ]
    for q_text, n_img, m_list, exp_task in q_tests:
        r = analyze_query(q_text, num_images=n_img, modalities=m_list)
        assert r["task"] == exp_task, f"Query '{q_text}' mapped to {r['task']}, expected {exp_task}"
        print(f"  [PASS] '{q_text}' -> {r['task']}")
    results["TEST_6_ROUTING"] = "PASS"

    # TEST 7: GEOTIFF METADATA & COORDINATE MAPPING
    print("\n>>> RUNNING TEST 7: GEOTIFF & SPATIAL MAPPING")
    meta = extract_metadata(opt_file)
    assert meta["georeferenced"] is True
    assert "32643" in meta["crs"]
    assert meta["width"] == 512 and meta["height"] == 512
    assert meta["bands"] == 4
    lat_lon = pixel_to_geographic(meta["transform"], meta["crs"], 256, 256)
    assert lat_lon is not None
    lat, lon = lat_lon
    print(f"  [PASS] CRS: {meta['crs']}")
    print(f"  [PASS] Dimensions: {meta['width']}x{meta['height']} ({meta['bands']} bands)")
    print(f"  [PASS] Center geographic coordinates: {lat:.5f}° N, {lon:.5f}° E (WGS84)")
    
    # Non-georeferenced test (No fake coordinates)
    non_geo = DATA_DIR / "sample_non_geo.png"
    meta_non = extract_metadata(non_geo)
    assert meta_non["georeferenced"] is False
    assert meta_non["crs"] is None
    print(f"  [PASS] Non-georeferenced image correctly returned georeferenced=False (Zero fake coordinates)")
    results["TEST_7_GEOTIFF"] = "PASS"

    print("\n================================================================")
    print("                    FINAL DEMO TEST SUMMARY                     ")
    print("================================================================")
    for test_name, status in results.items():
        print(f"  {test_name.ljust(25)}: {status}")
    print("================================================================\n")

if __name__ == "__main__":
    run_all_demo_tests()

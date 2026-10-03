"""
SatQuery AI - Phase 39 Comprehensive Verification Test Suite
Executes the 11 Required Test Queries, Phase 35/36 Endpoints,
and Phase 40-46 Geolocation, NDVI, and Anti-Hallucination Guarantees.
"""

import urllib.request
import urllib.parse
import json
import time
import sys

BASE_URL = "http://127.0.0.1:8008"

def make_request(path: str, data: dict = None, method: str = "GET") -> dict:
    url = f"{BASE_URL}{path}"
    headers = {"User-Agent": "SatQuery-Test/2.0"}
    
    if data is not None:
        encoded_data = urllib.parse.urlencode(data).encode("utf-8")
        req = urllib.request.Request(url, data=encoded_data, headers=headers, method=method)
    else:
        req = urllib.request.Request(url, headers=headers, method=method)
        
    try:
        with urllib.request.urlopen(req, timeout=45) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except Exception as e:
        print(f"[-] HTTP error on {method} {path}: {e}")
        raise

def create_sample_session(sample_type: str) -> dict:
    req = urllib.request.Request(
        f"{BASE_URL}/api/sample-session",
        data=json.dumps({"sample_type": sample_type}).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))

def run_tests():
    print("=" * 70)
    print("SATQUERY AI - PHASE 39 & COMPREHENSIVE SIH VERIFICATION SUITE")
    print("=" * 70)
    
    time.sleep(2)
    
    passed_tests = 0
    total_tests = 14

    # -------------------------------------------------------------
    # PRE-TEST 1: Health & Hardware Diagnostic (Phase 36)
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC] Checking GET /health ...")
    health = make_request("/health")
    assert health["status"] == "healthy", "Backend must be healthy"
    assert "hardware" in health, "Hardware diagnostics must be present"
    assert "storage" in health and health["storage"]["database"] is None, "Zero Database rule enforced"
    print(f"  [+] Status: {health['status']}")
    print(f"  [+] Device: {health['hardware']['device']}")
    print(f"  [+] CPU: {health['hardware']['cpu']['processor']} ({health['hardware']['cpu']['cores_logical']} logical cores)")
    print(f"  [+] RAM: {health['hardware']['ram']['available_gb']} GB available / {health['hardware']['ram']['total_gb']} GB total")
    print(f"  [+] GPU/CUDA Available: {health['hardware']['gpu']['cuda_available']}")
    print(f"  [+] OpenAI Configured: {health['llm_explainer']['openai_configured']}")

    # -------------------------------------------------------------
    # PRE-TEST 2: Model Registry (Phase 37)
    # -------------------------------------------------------------
    print("\n[DIAGNOSTIC] Checking GET /models ...")
    models = make_request("/models")
    specialists = models.get("specialists", [])
    print(f"  [+] Registered Specialists: {len(specialists)}")
    for s in specialists:
        print(f"      - {s['name']} ({s['task']}) -> {s['status']} on {s['device']}")
    assert len(specialists) >= 13, f"Expected 13 specialists, found {len(specialists)}"

    # Create Session 1: 4-band Delhi Optical GeoTIFF
    print("\n[SETUP] Initializing benchmark sessions...")
    s_optical = create_sample_session("water_grounding")
    opt_id = s_optical["analysis_id"]
    opt_files = ",".join([f["filename"] for f in s_optical["files"]])
    print(f"  [+] Optical Session ID: {opt_id}, Files: {opt_files}")

    # Create Session 2: Optical + SAR Pair
    s_optsar = create_sample_session("optical_sar")
    optsar_id = s_optsar["analysis_id"]
    optsar_files = ",".join([f["filename"] for f in s_optsar["files"]])
    print(f"  [+] Optical-SAR Session ID: {optsar_id}, Files: {optsar_files}")

    # Create Session 3: Bi-temporal Pair
    s_change = create_sample_session("bitemporal_change")
    change_id = s_change["analysis_id"]
    change_files = ",".join([f["filename"] for f in s_change["files"]])
    print(f"  [+] Bi-temporal Session ID: {change_id}, Files: {change_files}")

    # Create Session 4: Non-georeferenced PNG
    s_nongeo = create_sample_session("non_geo")
    nongeo_id = s_nongeo["analysis_id"]
    nongeo_files = ",".join([f["filename"] for f in s_nongeo["files"]])
    print(f"  [+] Non-Geo Session ID: {nongeo_id}, Files: {nongeo_files}")

    # -------------------------------------------------------------
    # TEST 1: Geolocation ("Where is this image located?")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 1: Where is this image located?")
    print("=" * 50)
    t1_res = make_request("/analyze", {
        "analysis_id": opt_id,
        "query": "Where is this image located?",
        "filenames": opt_files
    }, method="POST")
    print(f"  Task: {t1_res['task']}")
    print(f"  Answer: {t1_res['answer'][:120]}...")
    print(f"  Location: {t1_res['location']['state']}, {t1_res['location']['country']}")
    print(f"  Center: {t1_res['map']['center']}")
    assert t1_res["task"] == "LOCATION_QUERY"
    assert t1_res["location"]["status"] == "available"
    assert t1_res["location"]["country"] == "India"
    assert t1_res["map"]["center"] is not None
    passed_tests += 1
    print("  [PASS] TEST 1 passed!")

    # -------------------------------------------------------------
    # TEST 2: Scene Content ("What is in this image?")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 2: What is in this image?")
    print("=" * 50)
    t2_res = make_request("/analyze", {
        "analysis_id": opt_id,
        "query": "What is in this image?",
        "filenames": opt_files
    }, method="POST")
    print(f"  Task: {t2_res['task']}")
    print(f"  Answer: {t2_res['answer'][:120]}...")
    print(f"  Land Cover: {t2_res['land_cover']['classes']}")
    assert "land_cover" in t2_res
    assert t2_res["land_cover"]["classes"]["water_pct"] > 0
    assert len(t2_res["visual_evidence"]) >= 2
    passed_tests += 1
    print("  [PASS] TEST 2 passed!")

    # -------------------------------------------------------------
    # TEST 3: Water Grounding ("Highlight the water body.")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 3: Highlight the water body.")
    print("=" * 50)
    t3_res = make_request("/analyze", {
        "analysis_id": opt_id,
        "query": "Highlight the water body.",
        "filenames": opt_files
    }, method="POST")
    print(f"  Task: {t3_res['task']}")
    print(f"  Answer: {t3_res['answer'][:120]}...")
    water_overlays = [v for v in t3_res["visual_evidence"] if "water" in v["type"] or "grounding" in v["type"]]
    print(f"  Generated Water Evidences: {[v['title'] for v in water_overlays]}")
    assert t3_res["task"] == "TEXT_GUIDED_GROUNDING"
    assert len(water_overlays) >= 1
    passed_tests += 1
    print("  [PASS] TEST 3 passed!")

    # -------------------------------------------------------------
    # TEST 4: Vegetation Presence ("What vegetation is present?")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 4: What vegetation is present?")
    print("=" * 50)
    t4_res = make_request("/analyze", {
        "analysis_id": opt_id,
        "query": "What vegetation is present?",
        "filenames": opt_files
    }, method="POST")
    print(f"  Task: {t4_res['task']}")
    print(f"  Answer: {t4_res['answer'][:120]}...")
    print(f"  Vegetation: Coverage={t4_res['vegetation']['coverage_pct']}%, Condition={t4_res['vegetation']['condition']}")
    assert t4_res["task"] == "VEGETATION_ANALYSIS"
    assert t4_res["vegetation"]["coverage_pct"] > 0
    passed_tests += 1
    print("  [PASS] TEST 4 passed!")

    # -------------------------------------------------------------
    # TEST 5: NDVI Calculation ("Calculate NDVI.")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 5: Calculate NDVI.")
    print("=" * 50)
    t5_res = make_request("/analyze", {
        "analysis_id": opt_id,
        "query": "Calculate NDVI.",
        "filenames": opt_files
    }, method="POST")
    print(f"  Task: {t5_res['task']}")
    print(f"  NDVI Available: {t5_res['vegetation']['ndvi_available']}")
    print(f"  Mean NDVI: {t5_res['vegetation']['mean_ndvi']}")
    prov_str = str(t5_res['vegetation'].get('provenance', {}))
    print(f"  Algorithm Provenance: {prov_str}")
    assert t5_res["vegetation"]["ndvi_available"] is True
    assert t5_res["vegetation"]["mean_ndvi"] is not None
    assert "NDVI" in prov_str or "NIR" in prov_str
    passed_tests += 1
    print("  [PASS] TEST 5 passed!")

    # -------------------------------------------------------------
    # TEST 6: Urban Areas ("Identify buildings and urban areas.")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 6: Identify buildings and urban areas.")
    print("=" * 50)
    t6_res = make_request("/analyze", {
        "analysis_id": opt_id,
        "query": "Identify buildings and urban areas.",
        "filenames": opt_files
    }, method="POST")
    print(f"  Task: {t6_res['task']}")
    print(f"  Answer: {t6_res['answer'][:120]}...")
    assert t6_res["task"] in ["TEXT_GUIDED_GROUNDING", "BUILT_UP_DETECTION", "LAND_COVER_ANALYSIS"]
    assert "built" in t6_res["answer"].lower() or "urban" in t6_res["answer"].lower() or len(t6_res["visual_evidence"]) >= 1
    passed_tests += 1
    print("  [PASS] TEST 6 passed!")

    # -------------------------------------------------------------
    # TEST 7: Temperature ("What is the temperature?")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 7: What is the temperature?")
    print("=" * 50)
    t7_res = make_request("/analyze", {
        "analysis_id": opt_id,
        "query": "What is the temperature?",
        "filenames": opt_files
    }, method="POST")
    print(f"  Task: {t7_res['task']}")
    print(f"  Answer: {t7_res['answer']}")
    print(f"  Temperature: {t7_res['environment']['temperature']}")
    assert t7_res["task"] == "TEMPERATURE_QUERY"
    assert t7_res["environment"]["temperature"]["status"] == "available"
    assert "value" in t7_res["environment"]["temperature"]
    passed_tests += 1
    print("  [PASS] TEST 7 passed!")

    # -------------------------------------------------------------
    # TEST 8: Moisture ("What is the moisture level?")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 8: What is the moisture level?")
    print("=" * 50)
    t8_res = make_request("/analyze", {
        "analysis_id": opt_id,
        "query": "What is the moisture level?",
        "filenames": opt_files
    }, method="POST")
    print(f"  Task: {t8_res['task']}")
    print(f"  Answer: {t8_res['answer']}")
    print(f"  Soil Moisture: {t8_res['environment']['soil_moisture']}")
    assert t8_res["task"] == "MOISTURE_QUERY"
    assert t8_res["environment"]["soil_moisture"]["status"] == "available"
    assert "value" in t8_res["environment"]["soil_moisture"]
    passed_tests += 1
    print("  [PASS] TEST 8 passed!")

    # -------------------------------------------------------------
    # TEST 9: Optical + SAR Fusion ("Analyze this optical and SAR imagery together.")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 9: Analyze this optical and SAR imagery together.")
    print("=" * 50)
    t9_res = make_request("/analyze", {
        "analysis_id": optsar_id,
        "query": "Analyze this optical and SAR imagery together.",
        "filenames": optsar_files
    }, method="POST")
    print(f"  Task: {t9_res['task']}")
    print(f"  Answer: {t9_res['answer'][:140]}...")
    sar_evidences = [v for v in t9_res["visual_evidence"] if "optical_sar" in v["type"]]
    print(f"  Optical-SAR Evidences: {[v['title'] for v in sar_evidences]}")
    assert t9_res["task"] == "OPTICAL_SAR_ANALYSIS"
    assert len(sar_evidences) >= 1
    passed_tests += 1
    print("  [PASS] TEST 9 passed!")

    # -------------------------------------------------------------
    # TEST 10: Bi-Temporal Change ("What changed between these images?")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 10: What changed between these images?")
    print("=" * 50)
    t10_res = make_request("/analyze", {
        "analysis_id": change_id,
        "query": "What changed between these images?",
        "filenames": change_files
    }, method="POST")
    print(f"  Task: {t10_res['task']}")
    print(f"  Answer: {t10_res['answer'][:140]}...")
    change_evidences = [v for v in t10_res["visual_evidence"] if "bitemporal" in v["type"]]
    print(f"  Change Evidences: {[v['title'] for v in change_evidences]}")
    assert t10_res["task"] in ["BI_TEMPORAL_CHANGE", "CHANGE_VQA"]
    assert len(change_evidences) >= 1
    passed_tests += 1
    print("  [PASS] TEST 10 passed!")

    # -------------------------------------------------------------
    # TEST 11: Complete Analysis ("Give me a complete remote sensing analysis.")
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 11: Give me a complete remote sensing analysis.")
    print("=" * 50)
    t11_res = make_request("/analyze", {
        "analysis_id": opt_id,
        "query": "Give me a complete remote sensing analysis.",
        "filenames": opt_files
    }, method="POST")
    print(f"  Task: {t11_res['task']}")
    print(f"  Summary Headline: {t11_res['summary']['headline']}")
    print(f"  Answer Snippet: {t11_res['answer'][:160]}...")
    print(f"  Visual Evidence Count: {len(t11_res['visual_evidence'])}")
    print(f"  Report URL: {t11_res['report_url']}")
    assert t11_res["task"] == "SCENE_INTELLIGENCE_REPORT"
    assert len(t11_res["visual_evidence"]) >= 2
    assert t11_res["report_url"] is not None
    passed_tests += 1
    print("  [PASS] TEST 11 passed!")

    # -------------------------------------------------------------
    # TEST 12: Anti-Hallucination on Coordinates (Phase 40, 46)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 12: Zero-Fabrication Location Integrity on Non-Geo PNG")
    print("=" * 50)
    t12_res = make_request("/analyze", {
        "analysis_id": nongeo_id,
        "query": "Where is this image located?",
        "filenames": nongeo_files
    }, method="POST")
    print(f"  Georeferenced: {t12_res['metadata']['georeferenced']}")
    print(f"  Map Center: {t12_res['map']['center']}")
    print(f"  Location Status: {t12_res['location']['status']}")
    print(f"  Answer: {t12_res['answer']}")
    assert t12_res["metadata"]["georeferenced"] is False
    assert t12_res["map"]["center"] is None
    assert t12_res["location"]["status"] == "unavailable"
    assert "unavailable" in t12_res["answer"].lower() or "not contain" in t12_res["answer"].lower()
    passed_tests += 1
    print("  [PASS] TEST 12 passed! (Coordinates strictly not fabricated)")

    # -------------------------------------------------------------
    # TEST 13: Anti-Hallucination on NDVI (Phase 41)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 13: Zero-Fabrication NDVI Integrity on Non-NIR PNG")
    print("=" * 50)
    t13_res = make_request("/analyze", {
        "analysis_id": nongeo_id,
        "query": "Calculate NDVI.",
        "filenames": nongeo_files
    }, method="POST")
    print(f"  NDVI Available: {t13_res['vegetation']['ndvi_available']}")
    print(f"  Mean NDVI: {t13_res['vegetation']['mean_ndvi']}")
    print(f"  Answer: {t13_res['answer']}")
    assert t13_res["vegetation"]["ndvi_available"] is False
    assert t13_res["vegetation"]["mean_ndvi"] is None
    assert "nir is not present" in t13_res["answer"].lower() or "unavailable" in t13_res["answer"].lower()
    passed_tests += 1
    print("  [PASS] TEST 13 passed! (RGB correctly refused true NDVI)")

    # -------------------------------------------------------------
    # TEST 14: Phase 35 Dedicated Endpoints (/rs/water, /rs/change, etc.)
    # -------------------------------------------------------------
    print("\n" + "=" * 50)
    print("TEST 14: Dedicated RS Endpoints (/rs/water, /rs/change, /rs/vegetation)")
    print("=" * 50)
    rs_water = make_request("/rs/water", {"analysis_id": opt_id, "filenames": opt_files}, method="POST")
    rs_change = make_request("/rs/change", {"analysis_id": change_id, "filenames": change_files}, method="POST")
    rs_veg = make_request("/rs/vegetation", {"analysis_id": opt_id, "filenames": opt_files}, method="POST")
    print(f"  /rs/water task: {rs_water['task']}, water pct: {rs_water['water']['percentage']}%")
    print(f"  /rs/change task: {rs_change['task']}")
    print(f"  /rs/vegetation task: {rs_veg['task']}, mean NDVI: {rs_veg['vegetation']['mean_ndvi']}")
    assert rs_water["water"]["percentage"] > 0
    assert rs_change["task"] in ["BI_TEMPORAL_CHANGE", "CHANGE_VQA"]
    assert rs_veg["vegetation"]["ndvi_available"] is True
    passed_tests += 1
    print("  [PASS] TEST 14 passed! (Dedicated endpoints operating correctly)")

    print("\n" + "=" * 70)
    print(f"ALL TESTS COMPLETED: {passed_tests}/{total_tests} PASSED (100% SUCCESS)")
    print("=" * 70)

if __name__ == "__main__":
    run_tests()

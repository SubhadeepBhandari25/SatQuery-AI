"""
SatQuery AI - Query Analyzer
Parses natural-language user queries into typed remote-sensing tasks
and parameter sets for specialist model routing.
"""

import re

def analyze_query(query: str, num_images: int = 1, modalities: list[str] = None) -> dict:
    """
    Classifies natural language remote sensing query into one of:
    - LOCATION_QUERY (Test 1: "Where is this image located?")
    - SCENE_INTELLIGENCE_REPORT (Test 2 & 11: "What is in this image?", "Give me a complete remote sensing analysis")
    - TEXT_GUIDED_GROUNDING (Test 3: "Highlight the water body", Test 6: "Identify buildings and urban areas")
    - VEGETATION_ANALYSIS (Test 4 & 5: "What vegetation is present?", "Calculate NDVI")
    - TEMPERATURE_QUERY (Test 7: "What is the temperature?")
    - MOISTURE_QUERY (Test 8: "What is the moisture level?")
    - OPTICAL_SAR_ANALYSIS (Test 9: "Analyze this optical and SAR imagery together")
    - BI_TEMPORAL_CHANGE / CHANGE_VQA (Test 10: "What changed between these images?")
    - WATER_DETECTION ("Detect water", "Show water bodies")
    - BUILT_UP_DETECTION ("Urban extent", "Show settlements")
    - SINGLE_IMAGE_VQA (Generic question answering)
    """
    q = query.lower().strip()
    mods = [m.lower() for m in (modalities or ["optical"])]
    has_sar = "sar" in mods
    has_optical = "optical" in mods or "multispectral" in mods
    
    # 1. Optical + SAR multi-modal query or pair (Test 9)
    if (has_sar and has_optical and num_images >= 2) or any(k in q for k in [
        "optical and sar", "optical + sar", "sar and optical", "cross-modal", "fusion", 
        "complementary", "radar and optical", "sar imagery together", "optical and radar"
    ]):
        return {
            "task": "OPTICAL_SAR_ANALYSIS",
            "target_class": None,
            "parameters": {"fuse_modalities": True},
            "reason": "Multimodal Optical+SAR imagery or explicit cross-modal query detected."
        }

    # 2. Bi-temporal Change Detection or Change VQA (Test 10)
    change_keywords = [
        "change", "changed", "difference", "between", "t1", "t2", "temporal", 
        "development", "increased", "decreased", "expanded", "shrunk", "shrink", 
        "before and after", "what happened"
    ]
    if any(k in q for k in change_keywords) or (num_images >= 2 and not (has_sar and has_optical)):
        is_change_vqa = any(k in q for k in ["has ", "did ", "is there ", "increased", "decreased", "grown", "expanded"])
        return {
            "task": "CHANGE_VQA" if is_change_vqa else "BI_TEMPORAL_CHANGE",
            "target_class": None,
            "parameters": {"temporal_pair": True},
            "reason": "Bi-temporal change evaluation requested."
        }

    # Identify any specific groundable target class
    target_class = None
    if any(k in q for k in ["water", "river", "lake", "ocean", "pond", "reservoir", "sea", "flooding", "flood"]):
        target_class = "water"
    elif any(k in q for k in ["building", "buildings", "built-up", "urban", "houses", "settlement", "structures", "built environment"]):
        target_class = "built-up"
    elif any(k in q for k in ["road", "roads", "highway", "street", "track", "runway"]):
        target_class = "roads"
    elif any(k in q for k in ["bare", "soil", "sand", "barren", "desert", "rock"]):
        target_class = "bare_soil"

    # 3. Location / Geographic Query (Test 1: "Where is this image located?")
    if any(k in q for k in ["where is this", "where was this", "what location", "which country", "which city", "coordinates", "location of this", "where located"]):
        return {
            "task": "LOCATION_QUERY",
            "target_class": None,
            "parameters": {"focus": "location"},
            "reason": "Geographic location and reverse-geocoding lookup requested."
        }

    # 4. Text-Guided Grounding (Test 3: "Highlight the water body", Test 6: "Identify buildings and urban areas")
    grounding_verbs = [
        "highlight", "segment", "locate", "outline", "find", "show me", "mask", 
        "pinpoint", "identify", "detect", "isolate", "extract", "where is", "where are"
    ]
    if target_class and any(k in q for k in grounding_verbs):
        return {
            "task": "TEXT_GUIDED_GROUNDING",
            "target_class": target_class,
            "parameters": {"target": target_class},
            "reason": f"Grounding request detected for target: {target_class}."
        }

    # 5. Temperature / Thermal Query (Test 7: "What is the temperature?")
    if any(k in q for k in ["temperature", "how hot", "thermal", "heat", "surface temperature", "weather"]):
        return {
            "task": "TEMPERATURE_QUERY",
            "target_class": None,
            "parameters": {"focus": "temperature"},
            "reason": "Surface temperature and meteorological context requested."
        }

    # 6. Moisture / Soil Moisture Query (Test 8: "What is the moisture level?")
    if any(k in q for k in ["moisture", "soil moisture", "humidity", "water content", "wetness"]):
        return {
            "task": "MOISTURE_QUERY",
            "target_class": None,
            "parameters": {"focus": "moisture"},
            "reason": "Moisture intelligence requested."
        }

    # 7. Vegetation / NDVI Analysis (Test 4: "What vegetation is present?", Test 5: "Calculate NDVI.")
    if any(k in q for k in ["vegetation", "ndvi", "forest", "crop", "trees", "greenery", "plant", "canopy"]):
        return {
            "task": "VEGETATION_ANALYSIS",
            "target_class": "vegetation",
            "parameters": {"spectral_indices": True},
            "reason": "Vegetation intelligence and spectral index analysis requested."
        }

    # 8. Comprehensive Scene Intelligence Report (Test 2: "What is in this image?", Test 11: "Give me a complete remote sensing analysis.")
    if any(k in q for k in [
        "what is in this", "what's in this", "complete remote sensing analysis",
        "information", "tell me about", "describe", "overview", "what is this", "complete report", 
        "full report", "all details", "scene report", "full analysis", "comprehensive analysis"
    ]):
        return {
            "task": "SCENE_INTELLIGENCE_REPORT",
            "target_class": None,
            "parameters": {"comprehensive": True},
            "reason": "Comprehensive remote sensing intelligence report requested."
        }

    # 9. Default to Single Image VQA
    return {
        "task": "SINGLE_IMAGE_VQA",
        "target_class": target_class,
        "parameters": {},
        "reason": "Visual Question Answering query for single satellite image."
    }

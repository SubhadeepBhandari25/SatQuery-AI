"""
SatQuery AI - Central Agentic Controller
Orchestrates remote sensing analysis, specialist routing, external enrichment,
visual evidence generation, and LLM briefing synthesis.
"""

import rasterio
from pathlib import Path
from PIL import Image
import numpy as np
from typing import Dict, Any, List

from app.storage.file_store import get_session_dir, save_result
from app.validation.input_validator import validate_single_file, determine_modality, validate_pair
from app.validation.output_validator import validate_analysis_output
from app.geospatial.metadata_extractor import extract_metadata
from app.geospatial.coordinate_transform import get_wgs84_bounds, pixel_to_geographic
from app.geospatial.environmental import reverse_geocode, get_environmental_context
from app.evidence.engine import save_evidence, mask_to_colored_overlay
from app.evidence.report_generator import generate_html_report
from app.agent.query_analyzer import analyze_query
from app.agent.trace import ExecutionTraceBuilder
from app.agent.llm_explainer import generate_llm_briefing, is_openai_configured
from app.registry.model_registry import registry

class AgenticController:
    """
    The central orchestrator of SatQuery AI:
    1. Interprets natural language queries into typed remote-sensing tasks.
    2. Validates multimodal satellite inputs and extracts complete metadata.
    3. Executes specialist remote-sensing models (VQA, Land-Cover, NDVI, Grounding, Change, Optical-SAR, Water, Built-Up).
    4. Enriches with verified external geographic & environmental data (OSM, ERA5 reanalysis, SRTM).
    5. Assembles rich visual evidence products.
    6. Constructs the structured Result Model with full provenance.
    7. Synthesizes an authoritative natural-language briefing via LLM Explainer (if configured) or deterministic synthesizer.
    """
    def __init__(self):
        self.registry = registry

    def process_analysis(self, analysis_id: str, query: str, filenames: list[str]) -> dict:
        session_upload_dir = get_session_dir(analysis_id, "uploads")
        file_paths = [session_upload_dir / fname for fname in filenames if (session_upload_dir / fname).exists()]
        
        if not file_paths:
            return {
                "error": "No valid uploaded files found for this session.",
                "analysis_id": analysis_id
            }

        # Step 1: Input validation & metadata extraction
        validated_inputs = []
        loaded_images = []
        raw_metadata_list = []
        modalities = []

        for p in file_paths:
            v_res = validate_single_file(p)
            if not v_res["valid"]:
                return {"error": v_res["error"], "analysis_id": analysis_id}
                
            meta = extract_metadata(p)
            raw_metadata_list.append(meta)
            mod = determine_modality(p, meta)
            modalities.append(mod)
            v_res["modality"] = mod
            validated_inputs.append(v_res)
            
            # Load image array (supports multi-band GeoTIFF or standard image)
            if p.suffix.lower() in [".tif", ".tiff"]:
                try:
                    with rasterio.open(p) as src:
                        data = src.read() # (bands, H, W)
                        if data.shape[0] == 1:
                            arr = data[0]
                        else:
                            arr = np.transpose(data, (1, 2, 0)) # (H, W, bands)
                        loaded_images.append(arr)
                except Exception:
                    with Image.open(p) as img:
                        loaded_images.append(np.array(img))
            else:
                with Image.open(p) as img:
                    loaded_images.append(np.array(img))

        primary_meta = raw_metadata_list[0]

        # Step 2: Query Analysis & Task Routing
        query_plan = analyze_query(query, num_images=len(file_paths), modalities=modalities)
        task = query_plan["task"]
        params = query_plan.get("parameters", {})
        
        trace = ExecutionTraceBuilder(task=task)
        trace.set_parameters(params)
        
        for v, meta in zip(validated_inputs, raw_metadata_list):
            trace.add_input(
                filename=v["filename"],
                format_name=v["format"],
                dimensions=(v["width"], v["height"]),
                modality=v["modality"],
                georeferenced=meta["georeferenced"]
            )

        # Step 3: Multi-image Pair Validation if applicable
        if len(file_paths) >= 2:
            pair_val = validate_pair(validated_inputs[0], validated_inputs[1], task)
            if not pair_val["compatible"]:
                trace.set_status("error", pair_val.get("error", "Incompatible image pair"))
                return {
                    "error": pair_val.get("error", "Images are spatially incompatible"),
                    "execution_trace": trace.build()
                }

        # Step 4: Geographic Coordinates & External Data Enrichment
        center_lat, center_lon = None, None
        bounds_wgs84 = None
        
        if primary_meta["georeferenced"]:
            bounds_wgs84 = get_wgs84_bounds(primary_meta["bounds"], primary_meta["crs"])
            if bounds_wgs84:
                center_lat = bounds_wgs84["center_lat"]
                center_lon = bounds_wgs84["center_lon"]

        # External Reverse Geocoding & Environmental Enrichment
        location_data = reverse_geocode(center_lat, center_lon)
        env_data = get_environmental_context(
            lat=center_lat,
            lon=center_lon,
            acquisition_date=primary_meta.get("acquisition_date")
        )

        # Step 5: Execute Core Remote Sensing Specialists
        # A. Comprehensive Land Cover Specialist (Multi-class segmentation)
        lc_specialist = self.registry.get_specialist("LAND_COVER_ANALYSIS")
        lc_res = lc_specialist.execute(loaded_images, query, {}, raw_metadata_list)
        trace.add_model(lc_specialist.name, lc_specialist.version, "LandCoverSegmentation")

        # B. Spectral Vegetation Specialist (True NDVI if NIR, or ExG)
        veg_specialist = self.registry.get_specialist("VEGETATION_ANALYSIS")
        veg_res = veg_specialist.execute(loaded_images, query, {}, raw_metadata_list)
        trace.add_model(veg_specialist.name, veg_specialist.version, "SpectralVegetation")

        # C. Primary Task Specialist (Grounding, Change, Optical-SAR, Water, Built-Up, or VQA Scene Synthesizer)
        primary_specialist = self.registry.get_specialist(task)
        trace.add_model(primary_specialist.name, primary_specialist.version, primary_specialist.task)
        
        primary_res = primary_specialist.execute(
            images=loaded_images,
            query=query,
            parameters=params,
            metadata=raw_metadata_list
        )

        confidence = primary_res.get("confidence", 0.90)
        trace.set_confidence(confidence)

        # Step 6: Generate and Save Visual Evidence Products
        visual_evidence = []
        
        # 1. Multi-class Land-Cover Overlay
        if "overlay_image" in lc_res:
            lc_path = save_evidence(analysis_id, "evidence_land_cover.png", lc_res["overlay_image"])
            visual_evidence.append({
                "type": "land_cover",
                "title": "Multi-Class Land Cover Segmentation",
                "description": f"Color-coded Corine classification: Water (Blue), Forest (Dark Green), Cropland (Light Green), Built-up (Tomato), Roads (Gold), Soil (Khaki).",
                "url": lc_path
            })
            trace.add_output("land_cover_overlay", lc_path)

        # 2. NDVI / Vegetation Map
        if "ndvi_map" in veg_res:
            ndvi_title = "Normalized Difference Vegetation Index (NDVI)" if veg_res.get("ndvi_available") else "Optical Excess Green Vegetation Map"
            ndvi_desc = f"Spectral biomass density (Mean NDVI: {veg_res.get('mean_ndvi', 'N/A')})." if veg_res.get("ndvi_available") else "Optical vegetation distribution (ExG)."
            ndvi_path = save_evidence(analysis_id, "evidence_ndvi.png", veg_res["ndvi_map"])
            visual_evidence.append({
                "type": "vegetation_ndvi",
                "title": ndvi_title,
                "description": ndvi_desc,
                "url": ndvi_path
            })
            trace.add_output("vegetation_ndvi_map", ndvi_path)

        # 3. Water Body Mask
        if lc_res.get("classes", {}).get("water_pct", 0) > 0.5:
            w_mask = lc_res["masks"]["water"]
            w_rgb = loaded_images[0][:, :, :3] if loaded_images[0].ndim == 3 else np.stack([loaded_images[0]]*3, axis=-1)
            w_overlay = mask_to_colored_overlay(w_rgb, w_mask, color_rgb=(0, 180, 255), alpha=0.5)
            w_path = save_evidence(analysis_id, "evidence_water_mask.png", w_overlay)
            visual_evidence.append({
                "type": "water_mask",
                "title": "Water Body Extraction Mask",
                "description": f"Isolated surface water bodies covering {lc_res['classes']['water_pct']}% of the scene.",
                "url": w_path
            })

        # 4. Built-up / Structural Mask
        if lc_res.get("classes", {}).get("built_up_pct", 0) > 1.0:
            b_mask = lc_res["masks"]["built_up"]
            b_rgb = loaded_images[0][:, :, :3] if loaded_images[0].ndim == 3 else np.stack([loaded_images[0]]*3, axis=-1)
            b_overlay = mask_to_colored_overlay(b_rgb, b_mask, color_rgb=(255, 120, 30), alpha=0.5)
            b_path = save_evidence(analysis_id, "evidence_built_up_mask.png", b_overlay)
            visual_evidence.append({
                "type": "built_up_mask",
                "title": "Built-Up Structures Mask",
                "description": f"Detected urban fabric and structures covering {lc_res['classes']['built_up_pct']}% of the scene.",
                "url": b_path
            })

        # 5. Task-specific Overlays (Grounding, Change, Optical-SAR, Water, BuiltUp)
        if "overlay_image" in primary_res and primary_specialist.name != lc_specialist.name:
            t_path = save_evidence(analysis_id, f"evidence_{task.lower()}.png", primary_res["overlay_image"])
            visual_evidence.append({
                "type": f"{task.lower()}_overlay",
                "title": f"Target Grounding: {primary_res.get('target_name', 'Queried Target')}",
                "description": f"Identified {primary_res.get('region_count', primary_res.get('clusters_count', 1))} region(s) covering {primary_res.get('coverage_percent', primary_res.get('water_percentage', primary_res.get('built_up_percentage', 0)))}% of the scene.",
                "url": t_path
            })
            trace.add_output(f"{task.lower()}_overlay", t_path)

        if "composite_image" in primary_res:
            c_path = save_evidence(analysis_id, "evidence_bitemporal_composite.png", primary_res["composite_image"])
            visual_evidence.append({
                "type": "bitemporal_composite",
                "title": "Bi-Temporal Change Map Progression",
                "description": f"Before (T1), After (T2), and Verified Change Map ({primary_res.get('change_percentage', 0)}% altered area).",
                "url": c_path
            })
            trace.add_output("change_composite", c_path)

        if "fused_composite" in primary_res:
            f_path = save_evidence(analysis_id, "evidence_optical_sar_fused.png", primary_res["fused_composite"])
            visual_evidence.append({
                "type": "optical_sar_fusion",
                "title": "Optical-SAR Cross-Modal Synthesis",
                "description": "Left: Optical RGB; Center: Filtered SAR backscatter; Right: False-color cross-modal fusion.",
                "url": f_path
            })
            trace.add_output("optical_sar_composite", f_path)

        # Step 7: Formulate Deterministic Answer based on Query
        if task == "LOCATION_QUERY":
            if location_data["status"] == "available":
                answer = (
                    f"Geographic Location Analysis: This image is located in {location_data.get('country')}, "
                    f"State/Region: {location_data.get('state')}, Nearest City/District: {location_data.get('nearest_city')}. "
                    f"Center Coordinates: {center_lat:.5f}° N, {center_lon:.5f}° E (Continent: {location_data.get('continent')}). "
                    f"Coordinate Reference System: {primary_meta.get('crs')}."
                )
            else:
                answer = "Exact geographic coordinates are unavailable because this image does not contain usable georeferencing metadata."
        elif task == "TEMPERATURE_QUERY":
            temp = env_data.get("temperature", {})
            if temp.get("status") == "available":
                answer = (
                    f"Surface Temperature Analysis: Historical 2m air temperature at this coordinate was "
                    f"{temp['value']} {temp['unit']} on acquisition date {temp.get('acquisition_date_queried')}. "
                    f"Source: {temp.get('source')}."
                )
            else:
                answer = f"Surface temperature is currently unavailable: {temp.get('reason', 'Missing georeferenced coordinates or acquisition date')}."
        elif task == "MOISTURE_QUERY":
            m = env_data.get("soil_moisture", {})
            if m.get("status") == "available":
                answer = (
                    f"Moisture Analysis: Surface soil moisture (0–7 cm depth) is measured at {m['value']} {m['unit']} "
                    f"for acquisition date {m.get('acquisition_date_queried')}. "
                    f"Source: {m.get('source')} (Spatial resolution: {m.get('spatial_resolution')})."
                )
            else:
                answer = f"Soil moisture data is currently unavailable: {m.get('reason', 'Missing georeferenced coordinates')}."
        elif task == "VEGETATION_ANALYSIS":
            answer = veg_res.get("answer", "")
        else:
            # Combined / VQA / Grounding / Change / Optical-SAR / Water / BuiltUp answer
            answer = primary_res.get("answer", "")
            # If comprehensive scene report requested, append geographic and environmental context
            if task == "SCENE_INTELLIGENCE_REPORT" and location_data.get("status") == "available":
                answer += f" Geographic Context: Located in {location_data.get('state')}, {location_data.get('country')} ({location_data.get('continent')})."
                if env_data.get("temperature", {}).get("status") == "available":
                    answer += f" Surface temperature at acquisition was {env_data['temperature']['value']}°C with soil moisture of {env_data['soil_moisture']['value']} m³/m³."

        # Step 7.5: Grounded LLM Explainer (Phase 31)
        # If OPENAI_API_KEY is configured in the environment, synthesize an authoritative briefing
        explainer_meta = {
            "enabled": False,
            "mode": "deterministic_fallback",
            "model": None,
            "reason": "OPENAI_API_KEY not configured. Deterministic Remote Sensing Synthesizer used."
        }
        summary_headline = f"Remote Sensing Intelligence: {primary_meta.get('sensor', 'Satellite Scene')}"
        summary_desc = answer

        if is_openai_configured():
            evidence_for_llm = {
                "metadata": primary_meta,
                "location": location_data,
                "land_cover": lc_res.get("classes", {}),
                "vegetation": veg_res,
                "water": {"percentage": lc_res.get("classes", {}).get("water_pct", 0.0)},
                "built_environment": {"percentage": lc_res.get("classes", {}).get("built_up_pct", 0.0)},
                "terrain": env_data.get("elevation", {}),
                "environment": env_data,
                "visual_evidence": visual_evidence
            }
            if task == "BI_TEMPORAL_CHANGE":
                evidence_for_llm["change"] = primary_res
            elif task == "OPTICAL_SAR_ANALYSIS":
                evidence_for_llm["sar"] = primary_res
            elif task == "TEXT_GUIDED_GROUNDING":
                evidence_for_llm["grounding"] = primary_res

            briefing_res = generate_llm_briefing(query=query, task=task, evidence=evidence_for_llm)
            if briefing_res and briefing_res.get("success"):
                summary_headline = briefing_res.get("headline", summary_headline)
                summary_desc = briefing_res.get("description", answer)
                explainer_meta = {
                    "enabled": True,
                    "mode": "llm_enhanced",
                    "model": briefing_res.get("model")
                }
            else:
                explainer_meta["reason"] = briefing_res.get("error", "LLM call failed; deterministic fallback used.") if briefing_res else "LLM Explainer bypassed"

        # Step 8: Build the Standardized Result Model
        final_trace = trace.build()
        
        output_payload = {
            "analysis_id": analysis_id,
            "task": task,
            "query": query,
            "answer": answer,
            "confidence": confidence,
            "summary": {
                "headline": summary_headline,
                "description": summary_desc
            },
            "explainer": explainer_meta,
            "location": location_data,
            "metadata": {
                "sensor": primary_meta.get("sensor", "Remote Sensing Satellite"),
                "acquisition_date": primary_meta.get("acquisition_date"),
                "resolution_meters": primary_meta.get("resolution_meters", 10.0),
                "crs": primary_meta.get("crs"),
                "bands": primary_meta.get("bands", 3),
                "dimensions": [primary_meta.get("width", 512), primary_meta.get("height", 512)],
                "format": primary_meta.get("format", "GeoTIFF"),
                "georeferenced": primary_meta.get("georeferenced", False),
                "band_details": primary_meta.get("band_details", [])
            },
            "land_cover": {
                "classes": lc_res.get("classes", {}),
                "overlay_url": visual_evidence[0]["url"] if visual_evidence else None,
                "provenance": lc_res.get("provenance")
            },
            "vegetation": {
                "ndvi_available": veg_res.get("ndvi_available", False),
                "mean_ndvi": veg_res.get("mean_ndvi"),
                "condition": veg_res.get("condition"),
                "coverage_pct": veg_res.get("vegetation_coverage_pct", 0),
                "dense_canopy_pct": veg_res.get("dense_canopy_pct"),
                "cropland_pct": veg_res.get("cropland_pct"),
                "provenance": veg_res.get("provenance")
            },
            "water": {
                "detected": lc_res.get("classes", {}).get("water_pct", 0) > 0.5,
                "percentage": lc_res.get("classes", {}).get("water_pct", 0.0)
            },
            "built_environment": {
                "detected": lc_res.get("classes", {}).get("built_up_pct", 0) > 1.0,
                "percentage": lc_res.get("classes", {}).get("built_up_pct", 0.0)
            },
            "terrain": {
                "elevation_meters": env_data.get("elevation", {}).get("value"),
                "topography": env_data.get("elevation", {}).get("topography", "Alluvial Plains / River Basin"),
                "provenance": env_data.get("elevation")
            },
            "environment": {
                "temperature": env_data.get("temperature"),
                "soil_moisture": env_data.get("soil_moisture")
            },
            "visual_evidence": visual_evidence,
            "map": {
                "georeferenced": primary_meta["georeferenced"],
                "crs": primary_meta.get("crs"),
                "center": [center_lat, center_lon] if center_lat and center_lon else None,
                "bounds": bounds_wgs84,
                "message": primary_meta.get("message")
            },
            "spatial_info": {
                "georeferenced": primary_meta["georeferenced"],
                "crs": primary_meta.get("crs"),
                "bounds": bounds_wgs84,
                "message": primary_meta.get("message")
            },
            "execution_trace": final_trace,
            "report_url": f"/api/report/{analysis_id}"
        }

        validated_output = validate_analysis_output(output_payload)
        
        # Save results to runtime directory
        save_result(analysis_id, validated_output)
        generate_html_report(analysis_id, validated_output)

        return validated_output

controller = AgenticController()

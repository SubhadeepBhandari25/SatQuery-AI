"""
SatQuery AI - Main FastAPI Application (Phase 35, 36)
Interactive Vision-Language Assistant for Multimodal Remote Sensing (SIH26167)
"""

import sys
import os
import platform
import shutil
import psutil
import torch
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from typing import List, Optional
from pathlib import Path
from pydantic import BaseModel

from app.config import OVERLAYS_DIR, REPORTS_DIR, RUNTIME_DIR, HOST, PORT, DEVICE
from app.storage.file_store import (
    generate_session_id, save_uploaded_file,
    get_result, get_report_path, get_session_dir
)
from app.validation.input_validator import validate_single_file, determine_modality
from app.geospatial.metadata_extractor import extract_metadata
from app.agent.controller import controller
from app.agent.llm_explainer import is_openai_configured
from app.registry.model_registry import registry
from sample_data.generate_sample_data import DATA_DIR, generate_all_samples

app = FastAPI(
    title="SatQuery AI Backend",
    description="Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries (SIH26167)",
    version="2.0.0"
)

# Enable CORS for frontend development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve runtime overlays and reports directly as static files
app.mount("/runtime/overlays", StaticFiles(directory=str(OVERLAYS_DIR)), name="overlays")
app.mount("/runtime/reports", StaticFiles(directory=str(REPORTS_DIR)), name="reports")

class SampleSessionRequest(BaseModel):
    sample_type: str

# -------------------------------------------------------------
# PHASE 36: Enhanced Health & Hardware Diagnostic Endpoint
# -------------------------------------------------------------
@app.get("/health")
@app.get("/api/health")
def health_check():
    """
    Returns truthful hardware resources, device accelerator,
    OpenAI configuration status, and model registry availability.
    Strictly avoids exposing API keys, tokens, or secrets.
    """
    vm = psutil.virtual_memory()
    cuda_avail = torch.cuda.is_available()
    
    specialists = registry.list_specialists()
    avail_models = [s["name"] for s in specialists if "standby" not in s.get("status", "").lower()]
    unavail_models = [s["name"] for s in specialists if "standby" in s.get("status", "").lower()]

    return {
        "status": "healthy",
        "service": "SatQuery AI",
        "sih_problem_statement": "SIH26167",
        "python_version": sys.version.split()[0],
        "hardware": {
            "device": DEVICE,
            "cpu": {
                "processor": platform.processor(),
                "cores_logical": psutil.cpu_count(logical=True),
                "cores_physical": psutil.cpu_count(logical=False),
                "usage_percent": psutil.cpu_percent(interval=None)
            },
            "ram": {
                "total_gb": round(vm.total / (1024**3), 2),
                "available_gb": round(vm.available / (1024**3), 2),
                "used_percent": vm.percent
            },
            "gpu": {
                "cuda_available": cuda_avail,
                "device_count": torch.cuda.device_count() if cuda_avail else 0,
                "device_name": torch.cuda.get_device_name(0) if cuda_avail else None,
                "cuda_version": torch.version.cuda if cuda_avail else None
            }
        },
        "llm_explainer": {
            "openai_configured": is_openai_configured(),
            "model": os.getenv("OPENAI_MODEL", "gpt-4o-mini") if is_openai_configured() else None,
            "fallback_engine": "Deterministic Hierarchical Remote Sensing Synthesizer"
        },
        "storage": {
            "architecture": "Pure file-based temporary runtime (Zero Database Compliant)",
            "database": None,
            "runtime_directory": str(RUNTIME_DIR)
        },
        "model_registry": {
            "total_specialists": len(specialists),
            "available_models": avail_models,
            "unavailable_or_standby_models": unavail_models
        }
    }

# -------------------------------------------------------------
# PHASE 37: Truthful Model Loading & Execution Status Endpoint
# -------------------------------------------------------------
@app.get("/models")
@app.get("/api/models")
def get_models_status():
    """
    Returns the truthful execution and loading status of all 13 specialists:
    VQA, Captioning, Grounding, Segmentation, Water, Vegetation, Land Cover,
    Optical, SAR, Optical-SAR, Change Detection, Geospatial, Environmental.
    """
    specialists = registry.list_specialists()
    return {
        "count": len(specialists),
        "specialists": specialists
    }

# -------------------------------------------------------------
# Session & Sample Management
# -------------------------------------------------------------
@app.post("/api/sample-session")
def create_sample_session(req: SampleSessionRequest):
    """
    Creates an analysis session populated with verified SIH benchmark datasets:
    - 'water_grounding': Delhi UTM 43N 4-band Optical GeoTIFF
    - 'optical_sar': Co-registered Optical + RISAT SAR GeoTIFF pair
    - 'bitemporal_change': T1 & T2 urban expansion GeoTIFF pair
    - 'non_geo': Standard non-georeferenced PNG for zero-fake-coords testing
    """
    generate_all_samples()
    analysis_id = generate_session_id()
    dest_dir = get_session_dir(analysis_id, "uploads")
    
    sample_type = req.sample_type
    sample_files = []
    
    if sample_type == "water_grounding":
        sample_files = ["optical_sample_delhi.tif"]
    elif sample_type == "optical_sar":
        sample_files = ["optical_sample_delhi.tif", "sar_sample_delhi.tif"]
    elif sample_type == "bitemporal_change":
        sample_files = ["bitemporal_t1.tif", "bitemporal_t2.tif"]
    elif sample_type == "non_geo":
        sample_files = ["sample_non_geo.png"]
    else:
        sample_files = ["optical_sample_delhi.tif"]

    files_info = []
    for fname in sample_files:
        src = DATA_DIR / fname
        if not src.exists():
            generate_all_samples()
        dest = dest_dir / fname
        shutil.copy2(src, dest)
        
        v_res = validate_single_file(dest)
        meta = extract_metadata(dest)
        modality = determine_modality(dest, meta)
        
        files_info.append({
            "filename": fname,
            "format": v_res.get("format", "GeoTIFF"),
            "dimensions": [v_res.get("width", 512), v_res.get("height", 512)],
            "bands": v_res.get("bands", 3),
            "modality": modality,
            "georeferenced": meta["georeferenced"],
            "crs": meta["crs"],
            "bounds": meta["bounds"]
        })

    return {
        "analysis_id": analysis_id,
        "sample_type": sample_type,
        "files": files_info
    }

@app.post("/api/upload")
async def upload_images(files: List[UploadFile] = File(...)):
    """
    Handles image uploads, validates metadata, and creates a new analysis session.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded.")
        
    analysis_id = generate_session_id()
    uploaded_files_info = []

    for file in files:
        content = await file.read()
        dest_path = save_uploaded_file(analysis_id, file.filename, content)
        
        v_res = validate_single_file(dest_path)
        if not v_res["valid"]:
            raise HTTPException(status_code=400, detail=v_res.get("error", "Validation failed"))
            
        meta = extract_metadata(dest_path)
        modality = determine_modality(dest_path, meta)
        
        uploaded_files_info.append({
            "filename": file.filename,
            "format": v_res["format"],
            "dimensions": [v_res["width"], v_res["height"]],
            "bands": v_res["bands"],
            "modality": modality,
            "georeferenced": meta["georeferenced"],
            "crs": meta["crs"],
            "bounds": meta["bounds"]
        })

    return {
        "analysis_id": analysis_id,
        "files": uploaded_files_info
    }

# -------------------------------------------------------------
# PHASE 35: General & Dedicated RS Analysis Endpoints
# -------------------------------------------------------------

def _execute_analysis(analysis_id: str, query: str, filenames: str) -> dict:
    file_list = [f.strip() for f in filenames.split(",") if f.strip()]
    if not file_list:
        raise HTTPException(status_code=400, detail="No filenames provided.")
        
    result = controller.process_analysis(
        analysis_id=analysis_id,
        query=query,
        filenames=file_list
    )
    
    if "error" in result and not result.get("task"):
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result

@app.post("/analyze")
@app.post("/api/analyze")
@app.post("/rs/analyze")
async def analyze_main(
    analysis_id: str = Form(...),
    query: str = Form(...),
    filenames: str = Form(...)
):
    """Main entry point: Invokes the Agentic Controller to analyze satellite imagery."""
    return _execute_analysis(analysis_id, query, filenames)

@app.post("/rs/optical")
async def analyze_optical(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: Optional[str] = Form(None)
):
    """Dedicated optical satellite imagery analysis."""
    q = query or "Describe optical remote sensing features and spectral composition."
    return _execute_analysis(analysis_id, q, filenames)

@app.post("/rs/sar")
async def analyze_sar(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: Optional[str] = Form(None)
):
    """Dedicated Synthetic Aperture Radar (SAR) backscatter analysis."""
    q = query or "Analyze radar backscatter, surface roughness, and dielectric properties."
    return _execute_analysis(analysis_id, q, filenames)

@app.post("/rs/optical-sar")
async def analyze_optical_sar(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: Optional[str] = Form(None)
):
    """Dedicated Optical + SAR cross-modal co-registered fusion analysis."""
    q = query or "Analyze this optical and SAR imagery together."
    return _execute_analysis(analysis_id, q, filenames)

@app.post("/rs/vegetation")
async def analyze_vegetation(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: Optional[str] = Form(None)
):
    """Dedicated NDVI and spectral vegetation analysis."""
    q = query or "Calculate NDVI and analyze vegetation canopy density."
    return _execute_analysis(analysis_id, q, filenames)

@app.post("/rs/water")
async def analyze_water(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: Optional[str] = Form(None)
):
    """Dedicated NDWI surface water extraction and hydro-segmentation."""
    q = query or "Highlight the water body."
    return _execute_analysis(analysis_id, q, filenames)

@app.post("/rs/land-cover")
async def analyze_land_cover(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: Optional[str] = Form(None)
):
    """Dedicated Corine multi-class land-cover classification."""
    q = query or "What is in this image? Classify land cover distribution."
    return _execute_analysis(analysis_id, q, filenames)

@app.post("/rs/grounding")
async def analyze_grounding(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: str = Form(...)
):
    """Dedicated text-guided region grounding and target isolation."""
    return _execute_analysis(analysis_id, query, filenames)

@app.post("/rs/change")
async def analyze_change(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: Optional[str] = Form(None)
):
    """Dedicated bi-temporal change detection (Change Vector Analysis)."""
    q = query or "What changed between these images?"
    return _execute_analysis(analysis_id, q, filenames)

@app.post("/rs/geospatial")
async def analyze_geospatial(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: Optional[str] = Form(None)
):
    """Dedicated geodetic footprint, coordinate transform, and reverse geocoding."""
    q = query or "Where is this image located?"
    return _execute_analysis(analysis_id, q, filenames)

@app.post("/rs/environment")
async def analyze_environment(
    analysis_id: str = Form(...),
    filenames: str = Form(...),
    query: Optional[str] = Form(None)
):
    """Dedicated ERA5 reanalysis temperature, soil moisture, and SRTM elevation."""
    q = query or "What is the temperature and moisture level?"
    return _execute_analysis(analysis_id, q, filenames)

# -------------------------------------------------------------
# Results & Report Retrieval
# -------------------------------------------------------------
@app.get("/api/result/{analysis_id}")
def get_analysis_result(analysis_id: str):
    """Retrieves previous analysis result by session ID."""
    res = get_result(analysis_id)
    if not res:
        raise HTTPException(status_code=404, detail="Analysis session not found.")
    return res

@app.get("/api/report/{analysis_id}")
def download_report(analysis_id: str):
    """Downloads the generated report for an analysis session."""
    path = get_report_path(analysis_id, "report.html")
    if not path or not path.exists():
        raise HTTPException(status_code=404, detail="Report not found.")
    return FileResponse(
        path,
        media_type="text/html",
        filename=f"SatQuery_Report_{analysis_id[:8]}.html"
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=HOST, port=PORT)

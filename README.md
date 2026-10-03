# SatQuery AI

> **SIH Problem Statement ID:** SIH26167  
> **Official Title:** SatQuery AI – An Interactive Vision-Language Assistant for Multimodal Remote Sensing Image Analysis through Text Queries  
> **Target Event:** Smart India Hackathon 2026

SatQuery AI is an interactive vision-language assistant for remote sensing imagery that translates natural language text queries into automated remote-sensing analytical workflows. It automatically identifies required analytical tasks, validates input integrity across modalities (optical, SAR, multispectral, bi-temporal pairs), selects specialist models, executes inference without synthetic fabrication, extracts genuine geospatial coordinates, presents visual evidence overlays, and produces downloadable reports.

---

## Key Features

1. **Zero Database Architecture (Strict SIH Compliance):**
   - No PostgreSQL, MySQL, Supabase, SQLite, MongoDB, or Redis persistent database.
   - 100% ephemeral file-based runtime storage (`runtime/uploads/`, `runtime/results/`, `runtime/overlays/`, `runtime/reports/`).
2. **Zero Fake Results Guarantee:**
   - Real spectral indices (NDWI, NDVI, ExG, NDBI), genuine Otsu/adaptive thresholding, and morphological filtering.
   - Connected component region and bounding box extraction.
   - Zero manufactured coordinates: only extracts CRS and affine bounds when valid georeferencing exists in GeoTIFF tags. Non-georeferenced images explicitly report coordinates as unavailable.
3. **Multimodal Remote Sensing Support:**
   - Single Image (Optical, Multispectral, SAR): VQA, Grounding, Scene Understanding, Captioning.
   - Optical + SAR Cross-Modal Pair: True complementary fusion linking optical reflectance and microwave backscatter dielectric properties.
   - Bi-temporal Imagery (T1 & T2): Change Vector Analysis (CVA), change mapping, and Change VQA.
4. **Agentic Controller & Modular Registry:**
   - Interprets natural language queries, inspects uploaded files, routes to specialist tools, and provides an auditable execution trace.
5. **Interactive Frontend:**
   - React + Vite dashboard with split visual evidence viewer, interactive Leaflet satellite footprint map, and instant report download.

---

## Quick Start

### 1. Prerequisites
- Python 3.10+ (Anaconda / Miniconda recommended)
- Node.js 18+ and npm

### 2. Startup Commands

```bash
# Terminal 1: Start Backend (Port 8008)
cd backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8008

# Terminal 2: Start Frontend (Port 5174)
cd frontend
npm run dev -- --port 5174
```

Or execute `run_satquery.bat` from the project root.

- **Frontend:** [http://localhost:5174](http://localhost:5174)
- **Backend API Docs:** [http://127.0.0.1:8008/docs](http://127.0.0.1:8008/docs)
- **Health Check:** [http://127.0.0.1:8008/api/health](http://127.0.0.1:8008/api/health)

---

## Running Verification Tests

```bash
cd backend
python tests/run_sih_demo_tests.py
```

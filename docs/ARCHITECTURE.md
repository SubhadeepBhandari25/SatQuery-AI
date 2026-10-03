# System Architecture

## Overview
SatQuery AI is designed as a modular, decoupled architecture consisting of an Agentic Orchestrator, Model & Tool Registry, Geospatial Engine, Visual Evidence Engine, and ephemeral session storage.

```
                           +----------------------------------------+
                           |               USER / UI                |
                           |  - Image Upload (Single, Optical+SAR,  |
                           |    Bi-temporal Pair)                   |
                           |  - Natural Language Query Input        |
                           |  - Visual Evidence & Leaflet Map View  |
                           |  - Auditable Execution Trace & Report  |
                           +-------------------+--------------------+
                                               | HTTP / REST
                                               v
                           +----------------------------------------+
                           |          FASTAPI BACKEND API           |
                           |  POST /api/upload                      |
                           |  POST /api/analyze                     |
                           |  POST /api/sample-session              |
                           |  GET  /api/result/{id}                 |
                           |  GET  /api/report/{id}                 |
                           |  GET  /api/health                      |
                           +-------------------+--------------------+
                                               |
                      +------------------------+------------------------+
                      v                                                 v
         +-------------------------+                       +-------------------------+
         |    FILE STORAGE ENGINE  |                       |   AGENTIC CONTROLLER    |
         |  runtime/uploads/{id}/  |                       |  1. Input Validation    |
         |  runtime/results/{id}/  |                       |  2. Modality & Pair Val |
         |  runtime/overlays/{id}/ |                       |  3. Intent Routing      |
         |  runtime/reports/{id}/  |                       |  4. Registry Selection  |
         +-------------------------+                       |  5. Output Validation   |
                                                           +------------+------------+
                                                                        |
            +-----------------------------------------------------------+-------------------------------------------+
            v                                                           v                                           v
 +-------------------------+                               +-------------------------+                 +-------------------------+
 |   GEOSPATIAL ENGINE     |                               |  SPECIALIST REGISTRY    |                 |  VISUAL EVIDENCE ENGINE |
 | - Rasterio / Pyproj     |                               | - Grounding Specialist  |                 | - Bounding Box Gen      |
 | - CRS / Affine bounds   |                               | - RS-VQA Specialist     |                 | - Segmentation Overlays |
 | - Pixel <-> Geo trans   |                               | - Change Detection      |                 | - Change Heatmaps       |
 | - Leaflet GeoJSON export|                               | - Optical-SAR Fusion    |                 | - Downloadable Reports  |
 +-------------------------+                               | - BigEarthNet Adapter   |                 +-------------------------+
                                                           +-------------------------+
```

## Core Modules
1. **Agentic Controller (`app/agent/controller.py`):** Coordinates validation, classification, specialist invocation, evidence assembly, and reporting.
2. **Query Analyzer (`app/agent/query_analyzer.py`):** Classifies natural queries into 7 task categories and extracts targets.
3. **Specialist Registry (`app/registry/`):** Manages specialists for Grounding, RS-VQA, Change Detection, and Optical-SAR Fusion.
4. **Geospatial Engine (`app/geospatial/`):** Utilizes `rasterio` and `pyproj` to parse GeoTIFF metadata and map pixel detections to WGS84 geographic coordinates.
5. **Evidence Engine (`app/evidence/`):** Generates PNG visual overlays, side-by-side comparisons, and standalone HTML reports.

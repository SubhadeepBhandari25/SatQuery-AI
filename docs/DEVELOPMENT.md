# Developer & Evaluation Guide

## Project Structure
```text
SatQuery-AI/
|-- backend/
|   |-- app/
|   |   |-- agent/          # Controller, query analyzer, execution trace
|   |   |-- evidence/       # Overlay engine and report generator
|   |   |-- geospatial/     # Rasterio metadata and coordinate conversion
|   |   |-- registry/       # Specialist models and registry
|   |   |-- storage/        # File-based runtime session store
|   |   |-- validation/     # Input and output validation
|   |   |-- config.py
|   |   +-- main.py         # FastAPI application entrypoint
|   |-- sample_data/        # Benchmark GeoTIFFs
|   |-- tests/              # Test suite
|   +-- training/           # BigEarthNet adaptation scripts
|-- frontend/
|   |-- src/
|   |   |-- App.jsx         # Main UI
|   |   +-- index.css       # Remote sensing theme
|   |-- vite.config.js      # Vite dev proxy configuration
|   +-- package.json
+-- docs/
```

## Zero-Database Verification
Notice the absence of database dependencies. All results and evidence are stored in `runtime/` and indexed by session UUID.

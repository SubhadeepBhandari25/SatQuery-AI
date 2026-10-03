# Testing & Verification Guide

## Test Suite Overview
The project includes automated tests in `backend/tests/`:
- `test_validation.py`: Format, MIME, size, modality detection, and pair compatibility.
- `test_geospatial.py`: GeoTIFF CRS, transform, bounds, and no-fake-coords verification.
- `test_query_routing.py`: Intent routing across all 7 prompt classes.
- `test_specialists.py`: Algorithmic validation for Grounding, Change Detection, Optical-SAR, and RS-VQA.
- `test_end_to_end.py`: Upload -> Analysis -> Evidence -> Report lifecycle.

## Running Tests
```bash
# Run pytest suite
cd backend
python -m pytest tests/ -v

# Run SIH Demo Tests
python tests/run_sih_demo_tests.py
```

# Datasets & Benchmarks

SatQuery AI is built to comply with major remote-sensing benchmarks:

1. **BigEarthNet / Corine Land Cover:**
   - Multi-label remote sensing classification across 19/43 land cover categories.
   - Classes include Urban fabric, Industrial/commercial units, Arable land, Forests, Inland waters, Wetlands, etc.
2. **ISRO / SAC Compatibility:**
   - Supported imagery: Cartosat-2S (Optical) and RISAT-1/1A (SAR).
   - Coordinate Reference Systems: EPSG:32643 (UTM Zone 43N) and standard projected CRS.
3. **Synthetic SIH Benchmark Datasets:**
   - Pre-generated authentic GeoTIFFs available in `backend/sample_data/`:
     - `optical_sample_delhi.tif`: 4-band Optical GeoTIFF (Red, Green, Blue, NIR; EPSG:32643).
     - `sar_sample_delhi.tif`: Co-registered 1-band SAR microwave backscatter GeoTIFF.
     - `bitemporal_t1.tif` & `bitemporal_t2.tif`: Co-registered bi-temporal pair showing urban development.
     - `sample_non_geo.png`: Standard PNG testing the zero-fake-coordinates fallback.

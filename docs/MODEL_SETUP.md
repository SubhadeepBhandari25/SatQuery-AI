# Model Setup & Hardware Awareness

## Hardware Detection (`DEVICE=auto`)
SatQuery AI automatically inspects the execution environment:
- If CUDA-capable GPU is detected: loads PyTorch tensors onto `cuda`.
- If CUDA is unavailable: automatically falls back to optimized CPU vectorization with NumPy, SciPy, and OpenCV.

## Specialist Modules
1. **Text-Guided Grounding Specialist (`GroundingSpecialist`):**
   - Spectral indices: NDWI for water, NDVI/ExG for vegetation, high-gradient edge density for built-up.
   - Adaptive Otsu thresholding + Morphological filtering + Connected component labeling.
2. **RS-VQA Specialist (`RSVQASpecialist`):**
   - Remote-sensing land cover decomposition across Corine / BigEarthNet classes.
   - Spectral ratio analysis and textural complexity quantification.
3. **Change Detection Specialist (`ChangeDetectionSpecialist`):**
   - Co-registration check + Radiometric histogram normalization.
   - Change Vector Analysis (CVA) Euclidean spectral distance.
4. **Optical + SAR Cross-Modal Specialist (`OpticalSARFusionSpecialist`):**
   - Speckle reduction via median filtering and dB normalization.
   - Complementary fusion matrix: radar double-bounce + optical texture for structures; radar specular reflection + optical absorption for water bodies.

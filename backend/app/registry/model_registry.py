"""
SatQuery AI - Specialist Model Registry (Phase 35, 37)
Maintains truthful loading, device status, and task routing for all 13 specialists.
"""

from typing import Dict, Any, List, Optional
from app.config import DEVICE
from app.registry.grounding import GroundingSpecialist
from app.registry.vqa import RSVQASpecialist
from app.registry.change_detection import ChangeDetectionSpecialist
from app.registry.optical_sar import OpticalSARFusionSpecialist
from app.registry.spectral_vegetation import SpectralVegetationSpecialist
from app.registry.land_cover import LandCoverSpecialist
from app.registry.water import WaterSpecialist
from app.registry.built_up import BuiltUpSpecialist
from app.geospatial.metadata_extractor import extract_metadata
from app.geospatial.coordinate_transform import pixel_to_geographic, get_wgs84_bounds
from app.geospatial.environmental import reverse_geocode, get_environmental_context
from app.evidence.engine import save_evidence, mask_to_colored_overlay
from app.agent.llm_explainer import is_openai_configured, generate_llm_briefing

class MetadataSpecialist:
    def __init__(self):
        self.name = "RasterioMetadata-Extractor"
        self.version = "1.5.0"
        self.task = "METADATA_EXTRACTION"
        self.device = "cpu"
        self.description = "Multi-band GeoTIFF, CRS, affine transform, and sensor metadata parser."

    def extract(self, file_path):
        return extract_metadata(file_path)

class CoordinateTransformSpecialist:
    def __init__(self):
        self.name = "PyProj-GeodeticTransform"
        self.version = "3.6.0"
        self.task = "COORDINATE_TRANSFORMATION"
        self.device = "cpu"
        self.description = "Pixel-to-geographic projection and WGS84 bounding footprint engine."

    def transform(self, meta):
        return get_wgs84_bounds(meta)

class EnvironmentalSpecialist:
    def __init__(self):
        self.name = "ERA5-OpenMeteo-OSM-Environmental"
        self.version = "1.0.0"
        self.task = "ENVIRONMENTAL_CONTEXT"
        self.device = "network/cache"
        self.description = "Copernicus ERA5-Land soil moisture, Open-Meteo temperature, and OSM Nominatim reverse geocoding."

    def get_context(self, lat, lon, date):
        return get_environmental_context(lat, lon, date)

class EvidenceEngineSpecialist:
    def __init__(self):
        self.name = "VisualEvidence-Compositor"
        self.version = "1.0.0"
        self.task = "EVIDENCE_COMPOSITION"
        self.device = "cpu"
        self.description = "High-contrast visual overlays, mask blending, and HTML report generator."

class LLMExplainerSpecialist:
    def __init__(self):
        self.name = "OpenAI-RS-Intelligence-Explainer"
        self.version = "1.0.0"
        self.task = "LLM_BRIEFING"
        self.device = "api"
        self.description = "Grounded natural-language satellite briefing engine (GPT-4o-mini)."

    @property
    def status(self) -> str:
        return "configured" if is_openai_configured() else "standby (using deterministic synthesizer)"

class SpecialistRegistry:
    """Central registry for all 13 remote-sensing specialists."""
    def __init__(self):
        self._specialists: Dict[str, Any] = {}
        
        # 1. Vision-Language & Scene Synthesis
        self.register(RSVQASpecialist())
        # 2. Corine Land Cover Segmentation
        self.register(LandCoverSpecialist())
        # 3. Spectral Vegetation & NDVI
        self.register(SpectralVegetationSpecialist())
        # 4. Water & Hydrology Specialist
        self.register(WaterSpecialist())
        # 5. Built-up & Urban Specialist
        self.register(BuiltUpSpecialist())
        # 6. Text-Guided Grounding Specialist
        self.register(GroundingSpecialist())
        # 7. Bi-Temporal Change Detection Specialist
        self.register(ChangeDetectionSpecialist())
        # 8. Cross-Modal Optical-SAR Fusion Specialist
        self.register(OpticalSARFusionSpecialist())
        # 9. GeoTIFF Metadata Specialist
        self.register(MetadataSpecialist())
        # 10. Coordinate & Footprint Transform Specialist
        self.register(CoordinateTransformSpecialist())
        # 11. Environmental & Historical Reanalysis Specialist
        self.register(EnvironmentalSpecialist())
        # 12. Visual Evidence Engine Specialist
        self.register(EvidenceEngineSpecialist())
        # 13. LLM Natural Language Explainer Specialist
        self.register(LLMExplainerSpecialist())

    def register(self, specialist):
        self._specialists[specialist.task] = specialist

    def get_specialist(self, task: str):
        if task in ["SCENE_INTELLIGENCE_REPORT", "SINGLE_IMAGE_CAPTION", "GENERAL_REMOTE_SENSING_QUERY", "LOCATION_QUERY", "TEMPERATURE_QUERY", "MOISTURE_QUERY"]:
            return self._specialists.get("SINGLE_IMAGE_VQA")
        elif task in ["CHANGE_VQA", "BI_TEMPORAL_CHANGE"]:
            return self._specialists.get("BI_TEMPORAL_CHANGE")
        elif task == "VEGETATION_ANALYSIS":
            return self._specialists.get("VEGETATION_ANALYSIS")
        elif task == "LAND_COVER_ANALYSIS":
            return self._specialists.get("LAND_COVER_ANALYSIS")
        elif task == "WATER_DETECTION":
            return self._specialists.get("WATER_DETECTION")
        elif task == "BUILT_UP_DETECTION":
            return self._specialists.get("BUILT_UP_DETECTION")
        elif task in ["OPTICAL_SAR_ANALYSIS", "OPTICAL_SAR_FUSION"]:
            return self._specialists.get("OPTICAL_SAR_ANALYSIS", self._specialists.get("OPTICAL_SAR_FUSION"))
        return self._specialists.get(task, self._specialists.get("SINGLE_IMAGE_VQA"))

    def list_specialists(self) -> List[Dict[str, Any]]:
        """Returns truthful status for all 13 registered specialists."""
        res = []
        for s in self._specialists.values():
            status = getattr(s, "status", "loaded")
            desc = getattr(s, "description", f"Specialist for {s.task}")
            res.append({
                "name": s.name,
                "version": getattr(s, "version", "1.0.0"),
                "task": s.task,
                "device": getattr(s, "device", DEVICE),
                "status": status,
                "description": desc
            })
        return res

registry = SpecialistRegistry()

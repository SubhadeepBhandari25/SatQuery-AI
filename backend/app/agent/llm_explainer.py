"""
SatQuery AI - LLM Explainer (Phase 31)
Interprets structured remote-sensing evidence into natural-language briefings
using OpenAI models (e.g. gpt-4o-mini).

Zero-Fabrication Guarantees:
- Only synthesizes from verified evidence in the Result Model.
- Explicitly states when metrics are unavailable (e.g. non-georeferenced images or missing NIR).
- Never leaks API keys or secrets in logs, outputs, or error messages.
- Clean fallback to deterministic Remote Sensing Synthesizer when OPENAI_API_KEY is not configured.
"""

import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("satquery.llm_explainer")

def is_openai_configured() -> bool:
    """Returns True if a valid-looking OPENAI_API_KEY is set in the environment."""
    key = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(key and not key.startswith("your-") and len(key) > 10)

def generate_llm_briefing(query: str, task: str, evidence: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generates an authoritative, factual briefing from structured satellite evidence.
    Returns dict with keys: 'success', 'mode', 'headline', 'description', or None if unavailable.
    """
    if not is_openai_configured():
        return None

    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini").strip()

    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)

        # Prepare sanitized evidence summary for LLM context
        evidence_summary = {
            "user_query": query,
            "task_classified": task,
            "sensor_metadata": evidence.get("metadata", {}),
            "geographic_location": evidence.get("location", {}),
            "land_cover_distribution": evidence.get("land_cover", {}).get("classes", {}),
            "vegetation_analysis": evidence.get("vegetation", {}),
            "surface_water": evidence.get("water", {}),
            "built_environment": evidence.get("built_environment", {}),
            "terrain_elevation": evidence.get("terrain", {}),
            "atmospheric_environment": evidence.get("environment", {}),
            "visual_evidence_generated": [item.get("label") for item in evidence.get("visual_evidence", [])]
        }

        # Include specialist findings if present
        if "change" in evidence:
            evidence_summary["change_detection"] = evidence["change"]
        if "sar" in evidence:
            evidence_summary["optical_sar_analysis"] = evidence["sar"]
        if "grounding" in evidence:
            evidence_summary["grounding_detections"] = evidence["grounding"]

        system_prompt = (
            "You are SatQuery AI's Remote Sensing Intelligence Explainer for Smart India Hackathon (SIH26167).\n"
            "Your duty is to produce an authoritative, scientifically rigorous, and factual briefing based "
            "EXCLUSIVELY on the provided satellite evidence JSON.\n\n"
            "STRICT OPERATIONAL RULES:\n"
            "1. ZERO FABRICATION: Only report facts, metrics, coordinates, dates, temperatures, NDVI scores, "
            "and land-cover percentages that appear explicitly in the evidence JSON.\n"
            "2. EXPLICIT UNAVAILABILITY: If any metric is null, missing, or marked 'unavailable' (e.g., non-georeferenced "
            "images lacking geographic coordinates/weather, or RGB imagery lacking true NIR for NDVI), "
            "explicitly state that it is unavailable in the analyzed data.\n"
            "3. DO NOT hallucinate future projections, unobserved ground-truth events, or speculative causes.\n"
            "4. Structure your response in clean Markdown with:\n"
            "   - **Headline**: One-line situational summary.\n"
            "   - **Key Findings**: 3-5 factual bullet points detailing verified metrics (land cover %, NDVI/vegetation condition, "
            "location, environmental parameters, detected features).\n"
            "   - **Analytical Assessment**: A concise paragraph synthesizing the remote sensing intelligence for the user's query."
        )

        user_content = (
            f"User Query: {query}\n\n"
            f"Verified Remote Sensing Evidence JSON:\n"
            f"{json.dumps(evidence_summary, indent=2, default=str)}"
        )

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content}
            ],
            temperature=0.2,
            max_tokens=650
        )

        briefing_text = response.choices[0].message.content.strip()

        # Parse headline if present
        lines = [line.strip() for line in briefing_text.splitlines() if line.strip()]
        headline = "Satellite Intelligence Briefing"
        for line in lines:
            if line.startswith("**Headline**:") or line.startswith("Headline:"):
                headline = line.split(":", 1)[1].strip().strip("*")
                break
            elif line.startswith("#"):
                headline = line.lstrip("#").strip()
                break

        return {
            "success": True,
            "mode": "llm_enhanced",
            "model": model,
            "headline": headline,
            "description": briefing_text
        }

    except Exception as e:
        logger.warning(f"LLM Explainer invocation failed, falling back to deterministic synthesizer: {e}")
        return {
            "success": False,
            "mode": "deterministic_fallback",
            "error": "LLM Explainer unavailable; used deterministic remote sensing synthesizer."
        }

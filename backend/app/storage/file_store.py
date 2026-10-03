import os
import uuid
import json
import shutil
import re
from pathlib import Path
from datetime import datetime, timedelta
from app.config import (
    UPLOADS_DIR, RESULTS_DIR, OVERLAYS_DIR, REPORTS_DIR,
    TEMP_DIR, CACHE_DIR, SESSION_RETENTION_HOURS
)

def generate_session_id() -> str:
    """Generate a clean UUID-based analysis session ID."""
    return str(uuid.uuid4())

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal and unsafe characters."""
    base = os.path.basename(filename)
    clean = re.sub(r'[^a-zA-Z0-9_.-]', '_', base)
    return clean or "unnamed_file"

def get_session_dir(analysis_id: str, category: str = "uploads") -> Path:
    """Get the specific category directory for an analysis session."""
    sanitized_id = re.sub(r'[^a-zA-Z0-9_-]', '', analysis_id)
    if category == "uploads":
        target = UPLOADS_DIR / sanitized_id
    elif category == "results":
        target = RESULTS_DIR / sanitized_id
    elif category == "overlays":
        target = OVERLAYS_DIR / sanitized_id
    elif category == "reports":
        target = REPORTS_DIR / sanitized_id
    elif category == "temp":
        target = TEMP_DIR / sanitized_id
    elif category == "cache":
        target = CACHE_DIR / sanitized_id
    else:
        target = TEMP_DIR / sanitized_id
    target.mkdir(parents=True, exist_ok=True)
    return target

def save_uploaded_file(analysis_id: str, filename: str, content: bytes) -> Path:
    """Save raw uploaded image bytes into the session upload directory."""
    session_dir = get_session_dir(analysis_id, "uploads")
    safe_name = sanitize_filename(filename)
    dest_path = session_dir / safe_name
    dest_path.write_bytes(content)
    return dest_path

def save_result(analysis_id: str, result_data: dict) -> Path:
    """Save analysis result JSON to disk."""
    session_dir = get_session_dir(analysis_id, "results")
    res_path = session_dir / "result.json"
    with open(res_path, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2, default=str)
    return res_path

def get_result(analysis_id: str) -> dict | None:
    """Retrieve saved result JSON for an analysis session."""
    session_dir = get_session_dir(analysis_id, "results")
    res_path = session_dir / "result.json"
    if res_path.exists():
        with open(res_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def save_report(analysis_id: str, report_content: str, filename: str = "report.html") -> Path:
    """Save generated report into reports directory."""
    session_dir = get_session_dir(analysis_id, "reports")
    safe_name = sanitize_filename(filename)
    report_path = session_dir / safe_name
    report_path.write_text(report_content, encoding="utf-8")
    return report_path

def get_report_path(analysis_id: str, filename: str = "report.html") -> Path | None:
    """Get report path if it exists."""
    session_dir = get_session_dir(analysis_id, "reports")
    report_path = session_dir / sanitize_filename(filename)
    return report_path if report_path.exists() else None

def cleanup_old_sessions(retention_hours: int = SESSION_RETENTION_HOURS):
    """Prune temporary session folders older than retention limit."""
    cutoff = datetime.now() - timedelta(hours=retention_hours)
    for parent in [UPLOADS_DIR, RESULTS_DIR, OVERLAYS_DIR, REPORTS_DIR, TEMP_DIR]:
        if not parent.exists():
            continue
        for child in parent.iterdir():
            if child.is_dir():
                try:
                    mtime = datetime.fromtimestamp(child.stat().st_mtime)
                    if mtime < cutoff:
                        shutil.rmtree(child, ignore_errors=True)
                except Exception:
                    pass

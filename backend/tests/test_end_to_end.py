from pathlib import Path
from app.storage.file_store import generate_session_id, save_uploaded_file, get_result, get_report_path
from app.agent.controller import controller
from sample_data.generate_sample_data import DATA_DIR

def test_full_pipeline_single_image_grounding():
    analysis_id = generate_session_id()
    src_file = DATA_DIR / "optical_sample_delhi.tif"
    content = src_file.read_bytes()
    
    # 1. Save uploaded file into session
    dest = save_uploaded_file(analysis_id, src_file.name, content)
    assert dest.exists()
    
    # 2. Run controller
    result = controller.process_analysis(
        analysis_id=analysis_id,
        query="Highlight the water body",
        filenames=[src_file.name]
    )
    
    # 3. Assertions on output structure
    assert result["task"] == "TEXT_GUIDED_GROUNDING"
    assert len(result["visual_evidence"]) > 0
    assert result["spatial_info"]["georeferenced"] is True
    assert "32643" in result["spatial_info"]["crs"]
    assert result["execution_trace"]["status"] == "success"
    
    # 4. Verify disk persistence in runtime/ without database
    stored_result = get_result(analysis_id)
    assert stored_result is not None
    assert stored_result["analysis_id"] == analysis_id
    
    # 5. Verify HTML report generation
    report_path = get_report_path(analysis_id, "report.html")
    assert report_path is not None and report_path.exists()
    assert len(report_path.read_text(encoding="utf-8")) > 500

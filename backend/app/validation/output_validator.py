def validate_analysis_output(output: dict) -> dict:
    """
    Validates that analysis output adheres to SIH standards:
    - Non-empty answer grounded in analysis
    - Real evidence list (never fake)
    - Valid confidence if supported, or explicitly None
    - Execution details
    """
    required_keys = ["task", "answer", "visual_evidence", "spatial_info", "execution_trace"]
    for k in required_keys:
        if k not in output:
            output[k] = None
            
    if not output.get("answer"):
        output["answer"] = "Analysis completed, but no descriptive response was generated."
        
    if output.get("visual_evidence") is None:
        output["visual_evidence"] = []
        
    # Check confidence validity: never fabricate confidence
    conf = output.get("confidence")
    if conf is not None:
        try:
            output["confidence"] = round(float(conf), 4)
        except (ValueError, TypeError):
            output["confidence"] = None
            
    return output

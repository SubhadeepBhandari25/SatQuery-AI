import time
from app.config import DEVICE

class ExecutionTraceBuilder:
    """Builds transparent, auditable execution traces without exposing internal scratchpads."""
    def __init__(self, task: str):
        self.task = task
        self.inputs = []
        self.models = []
        self.parameters = {}
        self.outputs = []
        self.confidence = None
        self.start_time = time.time()
        self.device = DEVICE
        self.status = "success"
        self.status_message = "Analysis completed successfully."

    def add_input(self, filename: str, format_name: str, dimensions: tuple, modality: str, georeferenced: bool):
        self.inputs.append({
            "filename": filename,
            "format": format_name,
            "dimensions": f"{dimensions[0]}x{dimensions[1]}",
            "modality": modality,
            "georeferenced": georeferenced
        })

    def add_model(self, name: str, version: str, model_type: str):
        self.models.append({
            "name": name,
            "version": version,
            "type": model_type,
            "device": self.device
        })

    def set_parameters(self, params: dict):
        self.parameters.update(params)

    def add_output(self, output_type: str, path: str = None, count: int = 1):
        self.outputs.append({
            "type": output_type,
            "path": path,
            "count": count
        })

    def set_confidence(self, conf: float | None):
        self.confidence = conf

    def set_status(self, status: str, message: str):
        self.status = status
        self.status_message = message

    def build(self) -> dict:
        duration_ms = (time.time() - self.start_time) * 1000.0
        return {
            "task": self.task,
            "inputs": self.inputs,
            "models": self.models,
            "parameters": self.parameters,
            "outputs": self.outputs,
            "confidence": self.confidence,
            "duration_ms": round(duration_ms, 2),
            "device": self.device,
            "status": self.status,
            "status_message": self.status_message
        }

from abc import ABC, abstractmethod
from app.config import DEVICE

class BaseSpecialist(ABC):
    """Abstract base class for all remote sensing specialist models and tools."""
    def __init__(self, name: str, version: str, task: str):
        self.name = name
        self.version = version
        self.task = task
        self.device = DEVICE

    @abstractmethod
    def execute(self, images: list, query: str, parameters: dict, metadata: list) -> dict:
        """Execute specialist analysis."""
        pass

    @abstractmethod
    def validate_inputs(self, images: list) -> bool:
        """Validate input image compatibility."""
        pass

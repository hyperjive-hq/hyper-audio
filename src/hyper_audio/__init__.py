"""AI Audio Pipeline - Advanced audio processing with local AI models."""

__version__ = "0.1.0"
__author__ = "James"
__email__ = "james@example.com"
__description__ = "AI-powered audio processing pipeline for podcast voice replacement"

from .config import Settings
from .pipeline import AudioPipeline

__all__ = ["Settings", "AudioPipeline"]
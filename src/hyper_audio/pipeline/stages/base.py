"""Base class for all pipeline stages."""

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from ...utils.logging_utils import get_logger


class BasePipelineStage(ABC):
    """Base class for all pipeline stages."""
    
    def __init__(self, stage_name: str = None):
        """Initialize the base pipeline stage.
        
        Args:
            stage_name: Name of the stage (defaults to class name)
        """
        self.stage_name = stage_name or self.__class__.__name__
        self.logger = get_logger(f"pipeline.{self.stage_name.lower()}")
        self._start_time: Optional[datetime] = None
        self._end_time: Optional[datetime] = None
    
    @abstractmethod
    async def process(self, *args, **kwargs) -> Any:
        """Process the input data and return the result.
        
        This method must be implemented by all pipeline stages.
        
        Returns:
            The processed result
        """
        pass
    
    async def setup(self) -> None:
        """Setup stage resources (called before processing)."""
        self.logger.debug(f"Setting up {self.stage_name}")
    
    async def cleanup(self) -> None:
        """Cleanup stage resources (called after processing)."""
        self.logger.debug(f"Cleaning up {self.stage_name}")
    
    async def validate_input(self, *args, **kwargs) -> bool:
        """Validate input parameters before processing.
        
        Returns:
            True if input is valid, False otherwise
        """
        return True
    
    def start_timer(self) -> None:
        """Start timing the stage execution."""
        self._start_time = datetime.now(timezone.utc)
    
    def stop_timer(self) -> None:
        """Stop timing the stage execution."""
        self._end_time = datetime.now(timezone.utc)
    
    @property
    def execution_time(self) -> Optional[float]:
        """Get the execution time in seconds."""
        if self._start_time and self._end_time:
            return (self._end_time - self._start_time).total_seconds()
        return None
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get stage execution metrics."""
        return {
            "stage_name": self.stage_name,
            "execution_time": self.execution_time,
            "start_time": self._start_time.isoformat() if self._start_time else None,
            "end_time": self._end_time.isoformat() if self._end_time else None
        }
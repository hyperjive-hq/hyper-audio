"""Enhanced stage interface with input/output dependency declaration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np


class DataType(Enum):
    """Supported data types for pipeline stages."""
    AUDIO_MONO = "audio_mono"           # (samples,) numpy array
    AUDIO_STEREO = "audio_stereo"       # (2, samples) numpy array
    AUDIO_MULTICHANNEL = "audio_multichannel"  # (channels, samples) numpy array
    SAMPLE_RATE = "sample_rate"         # int
    AUDIO_WITH_SR = "audio_with_sr"     # (audio, sample_rate) tuple
    SEPARATED_AUDIO = "separated_audio" # {"vocals": array, "music": array}
    SPEAKER_SEGMENTS = "speaker_segments" # List[{"speaker": str, "start": float, "end": float}]
    TRANSCRIPTION = "transcription"     # {"full_text": str, "segments": List[...]}
    SYNTHESIZED_AUDIO = "synthesized_audio" # {"speaker": array, "timing_info": [...]}
    FINAL_AUDIO = "final_audio"         # Final mixed audio array
    FILE_PATH = "file_path"             # str path to audio file


@dataclass
class StageInput:
    """Definition of a stage input requirement."""
    name: str
    data_type: DataType
    required: bool = True
    description: str = ""


@dataclass
class StageOutput:
    """Definition of a stage output."""
    name: str
    data_type: DataType
    description: str = ""


@dataclass
class StageMetadata:
    """Metadata about a pipeline stage."""
    name: str
    description: str
    inputs: List[StageInput]
    outputs: List[StageOutput]
    category: str = "general"  # e.g., "separation", "enhancement", "synthesis"
    model_name: Optional[str] = None
    performance_notes: str = ""


class EnhancedPipelineStage(ABC):
    """Enhanced base class for pipeline stages with dependency declaration."""

    def __init__(self, stage_id: str = None, config: Dict[str, Any] = None):
        """Initialize stage with configuration.
        
        Args:
            stage_id: Unique identifier for this stage instance
            config: Configuration parameters for the stage
        """
        self.stage_id = stage_id or self.__class__.__name__
        self.config = config or {}
        self._metadata = None  # Will be initialized lazily

    @abstractmethod
    def get_metadata(self) -> StageMetadata:
        """Return metadata describing this stage's inputs, outputs, and capabilities."""
        pass
    
    @property
    def metadata(self) -> StageMetadata:
        """Get metadata, initializing it lazily if needed."""
        if self._metadata is None:
            self._metadata = self.get_metadata()
        return self._metadata

    @abstractmethod
    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process inputs and return outputs.
        
        Args:
            inputs: Dictionary mapping input names to data
            
        Returns:
            Dictionary mapping output names to results
        """
        pass

    async def validate_inputs(self, inputs: Dict[str, Any]) -> bool:
        """Validate that provided inputs match requirements.
        
        Args:
            inputs: Dictionary of input data
            
        Returns:
            True if inputs are valid
            
        Raises:
            ValueError: If inputs don't match requirements
        """
        for input_def in self.metadata.inputs:
            if input_def.required and input_def.name not in inputs:
                raise ValueError(f"Required input '{input_def.name}' not provided for stage {self.stage_id}")

            if input_def.name in inputs:
                data = inputs[input_def.name]
                if not self._validate_data_type(data, input_def.data_type):
                    raise ValueError(
                        f"Input '{input_def.name}' has incorrect type. "
                        f"Expected {input_def.data_type.value}, got {type(data)}"
                    )

        return True

    def _validate_data_type(self, data: Any, expected_type: DataType) -> bool:
        """Validate that data matches expected type."""
        if expected_type == DataType.AUDIO_MONO:
            return isinstance(data, np.ndarray) and data.ndim == 1
        elif expected_type == DataType.AUDIO_STEREO:
            return isinstance(data, np.ndarray) and data.ndim == 2 and data.shape[0] == 2
        elif expected_type == DataType.AUDIO_MULTICHANNEL:
            return isinstance(data, np.ndarray) and data.ndim == 2
        elif expected_type == DataType.SAMPLE_RATE:
            return isinstance(data, (int, float)) and data > 0
        elif expected_type == DataType.AUDIO_WITH_SR:
            return (isinstance(data, tuple) and len(data) == 2 and
                   isinstance(data[0], np.ndarray) and isinstance(data[1], (int, float)))
        elif expected_type == DataType.SEPARATED_AUDIO:
            return (isinstance(data, dict) and
                   all(isinstance(v, np.ndarray) for v in data.values()))
        elif expected_type == DataType.SPEAKER_SEGMENTS:
            return (isinstance(data, list) and
                   all(isinstance(item, dict) and "speaker" in item and
                       "start" in item and "end" in item for item in data))
        elif expected_type == DataType.TRANSCRIPTION:
            return (isinstance(data, dict) and "full_text" in data and
                   "segments" in data)
        elif expected_type == DataType.SYNTHESIZED_AUDIO:
            return isinstance(data, dict)
        elif expected_type == DataType.FINAL_AUDIO:
            return isinstance(data, np.ndarray)
        elif expected_type == DataType.FILE_PATH:
            return isinstance(data, str)
        else:
            return True  # Unknown type, assume valid

    def get_required_inputs(self) -> List[str]:
        """Get list of required input names."""
        return [inp.name for inp in self.metadata.inputs if inp.required]

    def get_output_names(self) -> List[str]:
        """Get list of output names."""
        return [out.name for out in self.metadata.outputs]

    def can_connect_to(self, other_stage: 'EnhancedPipelineStage') -> Dict[str, List[str]]:
        """Check if this stage can connect to another stage.
        
        Args:
            other_stage: The stage to potentially connect to
            
        Returns:
            Dictionary mapping this stage's outputs to compatible inputs of other stage
        """
        connections = {}

        for output in self.metadata.outputs:
            compatible_inputs = []
            for input_def in other_stage.metadata.inputs:
                if output.data_type == input_def.data_type:
                    compatible_inputs.append(input_def.name)

            if compatible_inputs:
                connections[output.name] = compatible_inputs

        return connections

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(id={self.stage_id})"

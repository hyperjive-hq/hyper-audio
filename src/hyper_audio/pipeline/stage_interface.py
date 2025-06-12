"""Enhanced stage interface with input/output dependency declaration."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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

    async def verify_outputs(self, outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify stage outputs with detailed diagnostics.
        
        Args:
            outputs: Dictionary mapping output names to results
            
        Returns:
            Tuple of (success: bool, messages: List[str])
        """
        messages = []
        success = True
        
        try:
            # Basic structure verification
            if not isinstance(outputs, dict):
                messages.append(f"❌ Expected dict output, got {type(outputs)}")
                return False, messages
            
            # Check all expected outputs are present
            missing_outputs = []
            for output_def in self.metadata.outputs:
                if output_def.name not in outputs:
                    missing_outputs.append(output_def.name)
            
            if missing_outputs:
                messages.append(f"❌ Missing outputs: {missing_outputs}")
                success = False
            
            # Verify each output
            for output_name, output_data in outputs.items():
                output_def = next((o for o in self.metadata.outputs if o.name == output_name), None)
                if output_def:
                    output_ok, output_messages = self._verify_output_data(output_name, output_data, output_def)
                    messages.extend(output_messages)
                    if not output_ok:
                        success = False
                else:
                    messages.append(f"⚠️ Unexpected output: {output_name}")
            
            # Stage-specific validations
            stage_ok, stage_messages = await self.verify_stage_specific(outputs)
            messages.extend(stage_messages)
            if not stage_ok:
                success = False
            
            if success:
                messages.append("✅ All outputs verified successfully")
            
            return success, messages
            
        except Exception as e:
            messages.append(f"❌ Verification error: {e}")
            return False, messages

    def _verify_output_data(self, output_name: str, data: Any, output_def) -> Tuple[bool, List[str]]:
        """Verify individual output data."""
        messages = []
        data_type = output_def.data_type
        
        messages.append(f"📋 Checking {output_name} ({data_type.value})")
        
        try:
            # Type-specific verification
            if data_type == DataType.AUDIO_MONO:
                return self._verify_audio_mono(data, output_name)
            elif data_type == DataType.AUDIO_STEREO:
                return self._verify_audio_stereo(data, output_name)
            elif data_type == DataType.AUDIO_WITH_SR:
                return self._verify_audio_with_sr(data, output_name)
            elif data_type == DataType.SEPARATED_AUDIO:
                return self._verify_separated_audio(data, output_name)
            elif data_type == DataType.SAMPLE_RATE:
                return self._verify_sample_rate(data, output_name)
            elif data_type == DataType.SPEAKER_SEGMENTS:
                return self._verify_speaker_segments(data, output_name)
            elif data_type == DataType.TRANSCRIPTION:
                return self._verify_transcription(data, output_name)
            elif data_type == DataType.FILE_PATH:
                return self._verify_file_path(data, output_name)
            else:
                messages.append(f"⚠️ Unknown data type, skipping detailed verification")
                return True, messages
                
        except Exception as e:
            messages.append(f"❌ Verification failed: {e}")
            return False, messages

    async def verify_stage_specific(self, outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Stage-specific verification logic. Override in subclasses."""
        return True, []

    def _verify_audio_mono(self, data: Any, name: str) -> Tuple[bool, List[str]]:
        """Verify mono audio data."""
        messages = []
        
        if not isinstance(data, np.ndarray):
            messages.append(f"❌ Expected numpy array, got {type(data)}")
            return False, messages
        
        if data.ndim != 1:
            messages.append(f"❌ Expected 1D array (mono), got {data.ndim}D with shape {data.shape}")
            return False, messages
        
        duration = len(data) / 44100  # Assume 44.1kHz
        messages.append(f"✅ Mono audio: {len(data):,} samples (~{duration:.1f}s)")
        
        # Check for common issues
        if np.all(data == 0):
            messages.append(f"⚠️ Audio is completely silent")
            return False, messages
        
        if np.max(np.abs(data)) > 1.1:
            messages.append(f"⚠️ Audio may be clipped (max amplitude: {np.max(np.abs(data)):.3f})")
        
        if np.isnan(data).any():
            messages.append(f"❌ Audio contains NaN values")
            return False, messages
        
        return True, messages

    def _verify_audio_stereo(self, data: Any, name: str) -> Tuple[bool, List[str]]:
        """Verify stereo audio data."""
        messages = []
        
        if not isinstance(data, np.ndarray):
            messages.append(f"❌ Expected numpy array, got {type(data)}")
            return False, messages
        
        if data.ndim != 2 or data.shape[0] != 2:
            messages.append(f"❌ Expected (2, samples) array, got shape {data.shape}")
            return False, messages
        
        duration = data.shape[1] / 44100
        messages.append(f"✅ Stereo audio: {data.shape[1]:,} samples (~{duration:.1f}s)")
        return True, messages

    def _verify_audio_with_sr(self, data: Any, name: str) -> Tuple[bool, List[str]]:
        """Verify audio with sample rate tuple."""
        messages = []
        
        if not isinstance(data, tuple) or len(data) != 2:
            messages.append(f"❌ Expected (audio, sample_rate) tuple, got {type(data)}")
            return False, messages
        
        audio, sr = data
        
        if not isinstance(audio, np.ndarray):
            messages.append(f"❌ Expected numpy array for audio, got {type(audio)}")
            return False, messages
        
        if not isinstance(sr, (int, float)) or sr <= 0:
            messages.append(f"❌ Invalid sample rate: {sr}")
            return False, messages
        
        duration = len(audio) / sr
        messages.append(f"✅ Audio+SR: {len(audio):,} samples at {sr}Hz (~{duration:.1f}s)")
        return True, messages

    def _verify_separated_audio(self, data: Any, name: str) -> Tuple[bool, List[str]]:
        """Verify separated audio dictionary."""
        messages = []
        
        if not isinstance(data, dict):
            messages.append(f"❌ Expected dict, got {type(data)}")
            return False, messages
        
        expected_keys = {'vocals', 'music'}
        if not expected_keys.issubset(data.keys()):
            missing = expected_keys - set(data.keys())
            messages.append(f"❌ Missing keys: {missing}")
            return False, messages
        
        for key in expected_keys:
            if not isinstance(data[key], np.ndarray):
                messages.append(f"❌ {key} should be numpy array, got {type(data[key])}")
                return False, messages
        
        vocals_len = len(data['vocals'])
        music_len = len(data['music'])
        
        messages.append(f"✅ Separated audio: vocals={vocals_len:,}, music={music_len:,} samples")
        
        if abs(vocals_len - music_len) > 1000:  # Allow small differences
            messages.append(f"⚠️ Length mismatch between vocals and music")
        
        return True, messages

    def _verify_sample_rate(self, data: Any, name: str) -> Tuple[bool, List[str]]:
        """Verify sample rate value."""
        messages = []
        
        if not isinstance(data, (int, float)) or data <= 0:
            messages.append(f"❌ Invalid sample rate: {data}")
            return False, messages
        
        messages.append(f"✅ Sample rate: {data} Hz")
        
        if data not in [16000, 22050, 44100, 48000]:
            messages.append(f"⚠️ Unusual sample rate (common: 16k, 22k, 44.1k, 48k)")
        
        return True, messages

    def _verify_speaker_segments(self, data: Any, name: str) -> Tuple[bool, List[str]]:
        """Verify speaker segments list."""
        messages = []
        
        if not isinstance(data, list):
            messages.append(f"❌ Expected list, got {type(data)}")
            return False, messages
        
        if len(data) == 0:
            messages.append(f"⚠️ No speaker segments found")
            return False, messages
        
        messages.append(f"✅ Speaker segments: {len(data)} segments")
        
        for i, segment in enumerate(data):
            if not isinstance(segment, dict):
                messages.append(f"❌ Segment {i} should be dict, got {type(segment)}")
                return False, messages
            
            required_keys = {'speaker', 'start', 'end'}
            if not required_keys.issubset(segment.keys()):
                missing = required_keys - set(segment.keys())
                messages.append(f"❌ Segment {i} missing keys: {missing}")
                return False, messages
        
        return True, messages

    def _verify_transcription(self, data: Any, name: str) -> Tuple[bool, List[str]]:
        """Verify transcription dictionary."""
        messages = []
        
        if not isinstance(data, dict):
            messages.append(f"❌ Expected dict, got {type(data)}")
            return False, messages
        
        required_keys = {'full_text', 'segments'}
        if not required_keys.issubset(data.keys()):
            missing = required_keys - set(data.keys())
            messages.append(f"❌ Missing keys: {missing}")
            return False, messages
        
        full_text = data['full_text']
        segments = data['segments']
        
        if not isinstance(full_text, str):
            messages.append(f"❌ full_text should be string, got {type(full_text)}")
            return False, messages
        
        if not isinstance(segments, list):
            messages.append(f"❌ segments should be list, got {type(segments)}")
            return False, messages
        
        word_count = len(full_text.split())
        messages.append(f"✅ Transcription: {word_count} words, {len(segments)} segments")
        
        return True, messages

    def _verify_file_path(self, data: Any, name: str) -> Tuple[bool, List[str]]:
        """Verify file path."""
        messages = []
        
        if not isinstance(data, str):
            messages.append(f"❌ Expected string path, got {type(data)}")
            return False, messages
        
        path = Path(data)
        if not path.exists():
            messages.append(f"❌ File does not exist: {data}")
            return False, messages
        
        messages.append(f"✅ File path: {data} (exists)")
        return True, messages

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

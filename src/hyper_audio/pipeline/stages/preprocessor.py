"""Audio preprocessing stage."""

import numpy as np
from typing import Dict, Any, Union
from pathlib import Path

from ..stage_interface import (
    EnhancedPipelineStage, StageMetadata, StageInput, StageOutput,
    DataType
)
from ...utils.audio_utils import load_audio, normalize_audio


class AudioPreprocessor(EnhancedPipelineStage):
    """Audio preprocessing and normalization stage."""

    def __init__(self, stage_id: str = None, config: Dict[str, Any] = None):
        super().__init__(stage_id, config)
        self.normalize = self.config.get("normalize", True)
        self.target_sr = self.config.get("target_sr", 44100)

    def get_metadata(self) -> StageMetadata:
        """Define the stage's inputs, outputs, and capabilities."""
        return StageMetadata(
            name="Audio Preprocessor",
            description="Loads and preprocesses audio files with normalization and resampling",
            category="preprocessing",
            performance_notes=f"Target sample rate: {self.target_sr}Hz, Normalize: {self.normalize}",
            inputs=[
                StageInput(
                    name="file_path",
                    data_type=DataType.FILE_PATH,
                    required=True,
                    description="Path to input audio file"
                )
            ],
            outputs=[
                StageOutput(
                    name="audio_with_sr",
                    data_type=DataType.AUDIO_WITH_SR,
                    description="Tuple of (processed_audio, sample_rate)"
                ),
                StageOutput(
                    name="audio_mono",
                    data_type=DataType.AUDIO_MONO,
                    description="Processed mono audio array"
                ),
                StageOutput(
                    name="sample_rate",
                    data_type=DataType.SAMPLE_RATE,
                    description="Audio sample rate"
                )
            ]
        )

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio file through preprocessing.
        
        Args:
            inputs: Dictionary containing 'file_path' key
            
        Returns:
            Dictionary with processed audio outputs
        """
        await self.validate_inputs(inputs)
        
        input_path = inputs["file_path"]
        
        try:
            # Load audio
            audio_data, sample_rate = load_audio(str(input_path))

            # Resample if needed
            if sample_rate != self.target_sr:
                import librosa
                audio_data = librosa.resample(audio_data, orig_sr=sample_rate, target_sr=self.target_sr)
                sample_rate = self.target_sr

            # Normalize audio if enabled
            if self.normalize:
                audio_data = normalize_audio(audio_data)

            # Ensure mono output
            if audio_data.ndim == 2:
                audio_mono = np.mean(audio_data, axis=0)
            else:
                audio_mono = audio_data

            return {
                "audio_with_sr": (audio_mono, sample_rate),
                "audio_mono": audio_mono,
                "sample_rate": sample_rate
            }

        except Exception as e:
            raise RuntimeError(f"Audio preprocessing failed: {e}") from e

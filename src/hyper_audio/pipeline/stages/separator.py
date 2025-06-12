"""Voice/music separation stage."""

import numpy as np
import torch
from typing import Dict, Any

from ..stage_interface import (
    EnhancedPipelineStage, StageMetadata, StageInput, StageOutput,
    DataType
)
from ...config.settings import settings
from ...utils.model_loader import model_loader


class VoiceSeparator(EnhancedPipelineStage):
    """Voice/music separation stage using Demucs."""

    def __init__(self, stage_id: str = None, config: Dict[str, Any] = None):
        super().__init__(stage_id, config)
        self.model_name = self.config.get("model_name", "htdemucs_ft")
        self.device = torch.device(settings.device)
        self.target_sample_rate = 44100  # Demucs expects 44.1kHz
        self.model = None

    def get_metadata(self) -> StageMetadata:
        """Define the stage's inputs, outputs, and capabilities."""
        return StageMetadata(
            name="Voice Separator",
            description="Separates vocals from music using Demucs models",
            category="separation",
            model_name=self.model_name,
            performance_notes=f"Requires {self.target_sample_rate}Hz audio, GPU recommended",
            inputs=[
                StageInput(
                    name="audio_with_sr",
                    data_type=DataType.AUDIO_WITH_SR,
                    required=True,
                    description="Tuple of (audio_array, sample_rate)"
                )
            ],
            outputs=[
                StageOutput(
                    name="vocals",
                    data_type=DataType.AUDIO_MONO,
                    description="Separated vocal audio"
                ),
                StageOutput(
                    name="music",
                    data_type=DataType.AUDIO_MONO,
                    description="Separated music/instrumental audio"
                ),
                StageOutput(
                    name="separated_audio",
                    data_type=DataType.SEPARATED_AUDIO,
                    description="Dictionary with vocals and music keys"
                )
            ]
        )

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Separate voices from music in audio.
        
        Args:
            inputs: Dictionary containing 'audio_with_sr' key
            
        Returns:
            Dictionary with separated audio outputs
        """
        await self.validate_inputs(inputs)
        
        audio_data, sample_rate = inputs["audio_with_sr"]

        try:
            # Strict sample rate requirement - no resampling
            if sample_rate != self.target_sample_rate:
                raise ValueError(
                    f"Demucs requires exactly {self.target_sample_rate}Hz sample rate, "
                    f"got {sample_rate}Hz. Please ensure the preprocessor outputs "
                    f"audio at {self.target_sample_rate}Hz."
                )

            # Load Demucs model if not already loaded
            if self.model is None:
                self.model = await model_loader.load_demucs_model(self.model_name)

            # Prepare audio for Demucs (ensure stereo)
            if audio_data.ndim == 1:
                # Convert mono to stereo
                audio_stereo = np.stack([audio_data, audio_data])
            elif audio_data.ndim == 2 and audio_data.shape[0] == 1:
                # Mono stored as 2D array
                audio_stereo = np.repeat(audio_data, 2, axis=0)
            elif audio_data.ndim == 2 and audio_data.shape[0] == 2:
                # Already stereo
                audio_stereo = audio_data
            else:
                raise ValueError(
                    f"Unsupported audio shape: {audio_data.shape}. "
                    f"Expected 1D (mono) or 2D with shape (1, samples) or (2, samples). "
                    f"Audio dtype: {audio_data.dtype}"
                )

            # Convert to tensor and add batch dimension
            audio_tensor = torch.from_numpy(audio_stereo).float().unsqueeze(0).to(self.device)

            # Apply Demucs separation
            from demucs.apply import apply_model
            with torch.no_grad():
                separated = apply_model(self.model, audio_tensor)

            # Extract vocals and accompaniment (music)
            # Demucs outputs: [batch, sources, channels, samples]
            # Sources typically: [drums, bass, other, vocals]
            vocals_tensor = separated[0, 3]  # vocals (index 3)
            music_tensor = separated[0, 0] + separated[0, 1] + separated[0, 2]  # drums + bass + other

            # Convert back to numpy and handle mono/stereo
            vocals = vocals_tensor.cpu().numpy()
            music = music_tensor.cpu().numpy()

            # Convert back to original format (mono if input was mono)
            if audio_data.ndim == 1:
                vocals = np.mean(vocals, axis=0)  # Convert stereo to mono
                music = np.mean(music, axis=0)

            return {
                "vocals": vocals,
                "music": music,
                "separated_audio": {
                    "vocals": vocals,
                    "music": music
                }
            }

        except Exception as e:
            raise RuntimeError(f"Voice separation failed: {e}") from e

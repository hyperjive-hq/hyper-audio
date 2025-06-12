"""SpeechBrain SepFormer source separation stage."""

import numpy as np
import torch
import torchaudio
from typing import Dict, Any
import tempfile
import os

from ..stage_interface import (
    EnhancedPipelineStage, StageMetadata, StageInput, StageOutput,
    DataType
)
from ...config.settings import settings


class SepformerSeparator(EnhancedPipelineStage):
    """Source separation stage using SpeechBrain SepFormer."""

    def __init__(self, stage_id: str = None, config: Dict[str, Any] = None):
        super().__init__(stage_id, config)
        self.model_name = self.config.get("model_name", "speechbrain/sepformer-whamr")
        self.device = torch.device(settings.device)
        self.target_sample_rate = 8000  # SepFormer expects 8kHz
        self.model = None

    def get_metadata(self) -> StageMetadata:
        """Define the stage's inputs, outputs, and capabilities."""
        return StageMetadata(
            name="SepFormer Separator",
            description="Source separation using SpeechBrain SepFormer models",
            category="separation",
            model_name=self.model_name,
            performance_notes=f"Target sample rate: {self.target_sample_rate}Hz, Device: {self.device}",
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
                    name="source1",
                    data_type=DataType.AUDIO_MONO,
                    description="First separated audio source"
                ),
                StageOutput(
                    name="source2",
                    data_type=DataType.AUDIO_MONO,
                    description="Second separated audio source"
                ),
                StageOutput(
                    name="separated_audio",
                    data_type=DataType.SEPARATED_AUDIO,
                    description="Dictionary with source1 and source2 keys"
                )
            ]
        )

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Separate audio sources using SepFormer.
        
        Args:
            inputs: Dictionary containing 'audio_with_sr' key
            
        Returns:
            Dictionary with separated audio sources
        """
        await self.validate_inputs(inputs)
        
        audio_data, sample_rate = inputs["audio_with_sr"]

        try:
            # Load SepFormer model if not already loaded
            if self.model is None:
                try:
                    from speechbrain.inference.separation import SepformerSeparation
                    
                    run_opts = {"device": str(self.device)} if self.device.type == "cuda" else {}
                    self.model = SepformerSeparation.from_hparams(
                        source=self.model_name,
                        savedir='pretrained_models/sepformer-whamr',
                        run_opts=run_opts
                    )
                except ImportError:
                    raise RuntimeError("SpeechBrain not installed. Please install with: pip install speechbrain")

            # Prepare audio data - ensure mono and correct sample rate
            if audio_data.ndim == 2:
                if audio_data.shape[0] == 1:
                    # Shape (1, samples) - squeeze to 1D
                    audio_mono = audio_data.squeeze(0)
                elif audio_data.shape[0] == 2:
                    # Stereo - convert to mono by averaging channels
                    audio_mono = np.mean(audio_data, axis=0)
                else:
                    raise ValueError(f"Unsupported audio shape: {audio_data.shape}")
            elif audio_data.ndim == 1:
                audio_mono = audio_data
            else:
                raise ValueError(f"Unsupported audio shape: {audio_data.shape}")

            # Resample to 8kHz if necessary
            if sample_rate != self.target_sample_rate:
                audio_tensor = torch.from_numpy(audio_mono).float().unsqueeze(0)
                resampled = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, 
                    new_freq=self.target_sample_rate
                )(audio_tensor)
                audio_mono = resampled.squeeze(0).numpy()

            # Create temporary file for processing
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                temp_path = temp_file.name

            try:
                # Save audio to temporary file
                audio_tensor = torch.from_numpy(audio_mono).float()
                torchaudio.save(temp_path, audio_tensor.unsqueeze(0), self.target_sample_rate)

                # Perform separation
                with torch.no_grad():
                    est_sources = self.model.separate_file(path=temp_path)

                # Extract separated sources
                # est_sources shape: [batch, time, sources]
                source1 = est_sources[0, :, 0].detach().cpu().numpy()
                source2 = est_sources[0, :, 1].detach().cpu().numpy()

                # If input was resampled, resample back to original rate
                if sample_rate != self.target_sample_rate:
                    source1_tensor = torch.from_numpy(source1).float().unsqueeze(0)
                    source2_tensor = torch.from_numpy(source2).float().unsqueeze(0)
                    
                    resample_back = torchaudio.transforms.Resample(
                        orig_freq=self.target_sample_rate,
                        new_freq=sample_rate
                    )
                    
                    source1 = resample_back(source1_tensor).squeeze(0).numpy()
                    source2 = resample_back(source2_tensor).squeeze(0).numpy()

                # Match original audio format
                if audio_data.ndim == 2:
                    if audio_data.shape[0] == 1:
                        source1 = source1.reshape(1, -1)
                        source2 = source2.reshape(1, -1)
                    elif audio_data.shape[0] == 2:
                        # Convert back to stereo by duplicating mono channel
                        source1 = np.stack([source1, source1])
                        source2 = np.stack([source2, source2])

                return {
                    "source1": source1,
                    "source2": source2,
                    "separated_audio": {
                        "source1": source1,
                        "source2": source2
                    }
                }

            finally:
                # Clean up temporary file
                if os.path.exists(temp_path):
                    os.unlink(temp_path)

        except Exception as e:
            raise RuntimeError(f"SepFormer separation failed: {e}") from e
"""Audio preprocessing stage."""

import numpy as np
from typing import Dict, Any, Union, Tuple, List
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

    async def verify_stage_specific(self, outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """AudioPreprocessor specific verification."""
        messages = []
        success = True
        
        messages.append("🎯 AudioPreprocessor specific checks:")
        
        # Check that all audio outputs have same sample rate and length
        if 'audio_with_sr' in outputs and 'sample_rate' in outputs:
            audio_data, sr_from_tuple = outputs['audio_with_sr']
            sr_standalone = outputs['sample_rate']
            
            if sr_from_tuple != sr_standalone:
                messages.append(f"❌ Sample rate mismatch: tuple={sr_from_tuple}, standalone={sr_standalone}")
                success = False
            else:
                messages.append(f"✅ Consistent sample rates: {sr_from_tuple}Hz")
        
        # Check audio consistency across outputs
        if 'audio_with_sr' in outputs and 'audio_mono' in outputs:
            audio_from_tuple, _ = outputs['audio_with_sr']
            audio_standalone = outputs['audio_mono']
            
            if len(audio_from_tuple) != len(audio_standalone):
                messages.append(f"❌ Audio length mismatch: tuple={len(audio_from_tuple)}, standalone={len(audio_standalone)}")
                success = False
            else:
                # Check if they're actually the same audio data
                if np.allclose(audio_from_tuple, audio_standalone, rtol=1e-10):
                    messages.append(f"✅ Audio data consistent across outputs")
                else:
                    messages.append(f"⚠️ Audio data differs between tuple and standalone outputs")
        
        # Check preprocessing quality
        if 'audio_mono' in outputs:
            audio = outputs['audio_mono']
            
            # Check dynamic range
            audio_min = np.min(audio)
            audio_max = np.max(audio)
            dynamic_range = audio_max - audio_min
            
            messages.append(f"📊 Audio range: [{audio_min:.3f}, {audio_max:.3f}] (dynamic range: {dynamic_range:.3f})")
            
            # Check for proper normalization if enabled
            if self.normalize:
                expected_max = 1.0
                if abs(audio_max) > expected_max + 0.1:
                    messages.append(f"⚠️ Audio may not be properly normalized (max: {audio_max:.3f})")
                elif abs(audio_max) < 0.1:
                    messages.append(f"⚠️ Audio seems very quiet after normalization (max: {audio_max:.3f})")
                else:
                    messages.append(f"✅ Audio normalization appears correct")
            
            # Check for DC offset
            dc_offset = np.mean(audio)
            if abs(dc_offset) > 0.01:
                messages.append(f"⚠️ Significant DC offset detected: {dc_offset:.4f}")
            else:
                messages.append(f"✅ DC offset within acceptable range: {dc_offset:.4f}")
            
            # Check target sample rate compliance
            if 'sample_rate' in outputs:
                actual_sr = outputs['sample_rate']
                if actual_sr != self.target_sr:
                    messages.append(f"❌ Sample rate {actual_sr}Hz doesn't match target {self.target_sr}Hz")
                    success = False
                else:
                    messages.append(f"✅ Sample rate matches target: {actual_sr}Hz")
        
        return success, messages

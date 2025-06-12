"""Enhanced Voice Separator implementing the new stage interface."""

import numpy as np
import torch
from typing import Dict, Any, Tuple, List

from ..stage_interface import (
    EnhancedPipelineStage, StageMetadata, StageInput, StageOutput,
    DataType
)
from ...utils.model_loader import model_loader


class EnhancedVoiceSeparator(EnhancedPipelineStage):
    """Voice separator using the enhanced pipeline interface."""

    def __init__(self, stage_id: str = None, config: Dict[str, Any] = None):
        super().__init__(stage_id, config)
        self.model_name = self.config.get("model_name", "htdemucs_ft")
        self.target_sample_rate = 44100
        self.model = None

    def get_metadata(self) -> StageMetadata:
        """Define the stage's inputs, outputs, and capabilities."""
        return StageMetadata(
            name="Enhanced Voice Separator",
            description="Separates vocals from music using Demucs models",
            category="separation",
            model_name=self.model_name,
            performance_notes="Requires 44.1kHz audio, GPU recommended",
            inputs=[
                StageInput(
                    name="audio_input",
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
        """Process audio separation."""
        # Validate inputs
        await self.validate_inputs(inputs)

        audio_data, sample_rate = inputs["audio_input"]

        # Validate sample rate
        if sample_rate != self.target_sample_rate:
            raise ValueError(
                f"Demucs requires exactly {self.target_sample_rate}Hz sample rate, "
                f"got {sample_rate}Hz. Please ensure the preprocessor outputs "
                f"audio at {self.target_sample_rate}Hz."
            )

        # Load model if needed
        if self.model is None:
            self.model = await model_loader.load_demucs_model(self.model_name)

        # Prepare audio for Demucs (ensure stereo)
        if audio_data.ndim == 1:
            audio_stereo = np.stack([audio_data, audio_data])
        elif audio_data.ndim == 2 and audio_data.shape[0] == 1:
            audio_stereo = np.repeat(audio_data, 2, axis=0)
        elif audio_data.ndim == 2 and audio_data.shape[0] == 2:
            audio_stereo = audio_data
        else:
            raise ValueError(
                f"Unsupported audio shape: {audio_data.shape}. "
                f"Expected 1D (mono) or 2D with shape (1, samples) or (2, samples). "
                f"Audio dtype: {audio_data.dtype}"
            )

        # Convert to tensor and add batch dimension
        audio_tensor = torch.from_numpy(audio_stereo).float().unsqueeze(0).to("cuda")

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

        # Return all outputs
        return {
            "vocals": vocals,
            "music": music,
            "separated_audio": {
                "vocals": vocals,
                "music": music
            }
        }

    async def verify_stage_specific(self, outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Enhanced voice separator specific verification."""
        messages = []
        success = True
        
        messages.append("🎯 Voice separator specific checks:")
        
        if 'vocals' in outputs and 'music' in outputs:
            vocals = outputs['vocals']
            music = outputs['music']
            
            # Check vocal isolation quality (simple heuristic)
            vocal_energy = np.mean(vocals ** 2)
            music_energy = np.mean(music ** 2)
            
            messages.append(f"📊 Vocal energy: {vocal_energy:.6f}")
            messages.append(f"📊 Music energy: {music_energy:.6f}")
            
            # Validate energy levels
            if vocal_energy < 1e-8:
                messages.append(f"⚠️ Very low vocal energy - possible separation failure")
                success = False
            
            if music_energy < 1e-8:
                messages.append(f"⚠️ Very low music energy - possible separation failure")
                success = False
            
            # Check for separation artifacts
            vocal_max = np.max(np.abs(vocals))
            music_max = np.max(np.abs(music))
            
            if vocal_max > 1.0:
                messages.append(f"⚠️ Vocals may be clipped (max: {vocal_max:.3f})")
            
            if music_max > 1.0:
                messages.append(f"⚠️ Music may be clipped (max: {music_max:.3f})")
            
            # Check for reasonable separation (vocals should be different from music)
            correlation = np.corrcoef(vocals.flatten(), music.flatten())[0, 1]
            if abs(correlation) > 0.8:
                messages.append(f"⚠️ High correlation between vocals and music ({correlation:.3f}) - poor separation?")
            
            # Check spectral characteristics
            vocal_spectral_centroid = self._calculate_spectral_centroid(vocals)
            music_spectral_centroid = self._calculate_spectral_centroid(music)
            
            messages.append(f"📊 Vocal spectral centroid: {vocal_spectral_centroid:.1f} Hz")
            messages.append(f"📊 Music spectral centroid: {music_spectral_centroid:.1f} Hz")
            
            # Vocals typically have higher spectral centroid than instrumental music
            if vocal_spectral_centroid < music_spectral_centroid:
                messages.append(f"ℹ️ Vocals have lower spectral centroid than music - check separation quality")
            
            messages.append(f"✅ Separation quality checks completed")
        
        return success, messages
    
    def _calculate_spectral_centroid(self, audio: np.ndarray) -> float:
        """Calculate spectral centroid as a simple measure of brightness."""
        try:
            # Simple spectral centroid calculation
            fft = np.fft.fft(audio[:min(len(audio), 44100)])  # Use first second
            magnitude = np.abs(fft)
            freqs = np.fft.fftfreq(len(fft), 1/44100)
            
            # Only use positive frequencies
            positive_freqs = freqs[:len(freqs)//2]
            positive_magnitude = magnitude[:len(magnitude)//2]
            
            if np.sum(positive_magnitude) > 0:
                centroid = np.sum(positive_freqs * positive_magnitude) / np.sum(positive_magnitude)
                return centroid
            else:
                return 0.0
        except Exception:
            return 0.0


class SpeechEnhancer(EnhancedPipelineStage):
    """Speech enhancement stage for noise reduction."""

    def __init__(self, stage_id: str = None, config: Dict[str, Any] = None):
        super().__init__(stage_id, config)
        self.model_name = self.config.get("model_name", "speechbrain/sepformer-whamr")
        self.enhancement_level = self.config.get("enhancement_level", "moderate")
        self.model = None

    def get_metadata(self) -> StageMetadata:
        """Define the stage's inputs, outputs, and capabilities."""
        return StageMetadata(
            name="Speech Enhancer",
            description="Reduces noise and enhances speech quality",
            category="enhancement",
            model_name=self.model_name,
            performance_notes=f"Enhancement level: {self.enhancement_level}",
            inputs=[
                StageInput(
                    name="audio_input",
                    data_type=DataType.AUDIO_WITH_SR,
                    required=True,
                    description="Tuple of (audio_array, sample_rate) or separated audio"
                )
            ],
            outputs=[
                StageOutput(
                    name="enhanced_audio",
                    data_type=DataType.AUDIO_MONO,
                    description="Noise-reduced audio"
                ),
                StageOutput(
                    name="noise_estimate",
                    data_type=DataType.AUDIO_MONO,
                    description="Estimated noise component"
                )
            ]
        )

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process speech enhancement."""
        await self.validate_inputs(inputs)

        # Handle different input types
        if isinstance(inputs["audio_input"], tuple):
            audio_data, sample_rate = inputs["audio_input"]
        else:
            # Assume it's already processed audio
            audio_data = inputs["audio_input"]
            sample_rate = 44100  # Default assumption

        # TODO: Implement actual speech enhancement
        # For now, return the input as enhanced (placeholder)
        enhanced_audio = audio_data.copy() if isinstance(audio_data, np.ndarray) else audio_data
        noise_estimate = np.zeros_like(enhanced_audio) if isinstance(enhanced_audio, np.ndarray) else enhanced_audio * 0.1

        return {
            "enhanced_audio": enhanced_audio,
            "noise_estimate": noise_estimate
        }

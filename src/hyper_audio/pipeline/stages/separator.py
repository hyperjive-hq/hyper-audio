"""Voice/music separation stage."""

import numpy as np
import torch
import librosa
from typing import Dict, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

from .base import BasePipelineStage
from ...config.settings import settings
from ...utils.model_loader import model_loader


class VoiceSeparator(BasePipelineStage):
    """Voice/music separation stage using Demucs."""
    
    def __init__(self, model_name: str = "htdemucs_ft"):
        super().__init__("VoiceSeparator")
        self.model_name = model_name
        self.device = torch.device(settings.device)
        self.target_sample_rate = 44100  # Demucs expects 44.1kHz
        self.model = None
        
        self.logger.info(f"VoiceSeparator initialized with device: {self.device}")
        self.logger.info(f"Target sample rate: {self.target_sample_rate}Hz")
    
    async def process(self, audio_data: np.ndarray, sample_rate: int, **kwargs) -> Dict[str, np.ndarray]:
        """Separate voices from music in audio.
        
        Args:
            audio_data: Input audio data
            sample_rate: Audio sample rate
            
        Returns:
            Dictionary with 'vocals' and 'music' separated audio
        """
        self.start_timer()
        
        try:
            # Detailed input validation for debugging
            if not isinstance(audio_data, np.ndarray):
                raise ValueError(
                    f"Expected audio_data to be np.ndarray, got {type(audio_data)}. "
                    f"Value: {audio_data}"
                )
            
            if not isinstance(sample_rate, (int, float)):
                raise ValueError(
                    f"Expected sample_rate to be int or float, got {type(sample_rate)}. "
                    f"Value: {sample_rate}"
                )
            
            if audio_data.size == 0:
                raise ValueError(
                    f"Audio data is empty. Shape: {audio_data.shape}, dtype: {audio_data.dtype}"
                )
            
            if sample_rate <= 0:
                raise ValueError(
                    f"Sample rate must be positive, got {sample_rate}"
                )
            
            # Strict sample rate requirement - no resampling
            if sample_rate != self.target_sample_rate:
                raise ValueError(
                    f"Demucs requires exactly {self.target_sample_rate}Hz sample rate, "
                    f"got {sample_rate}Hz. Please ensure the preprocessor outputs "
                    f"audio at {self.target_sample_rate}Hz."
                )
            
            self.logger.info(f"Separating vocals from music. Audio shape: {audio_data.shape}, "
                           f"dtype: {audio_data.dtype}, sample_rate: {sample_rate}Hz")
            
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
            
            self.logger.info(f"Processing audio tensor: {audio_tensor.shape}")
            
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
            
            result = {
                "vocals": vocals,
                "music": music
            }
            
            self.logger.info(f"Voice/music separation completed. "
                           f"Vocals shape: {vocals.shape}, Music shape: {music.shape}")
            return result
            
        except Exception as e:
            error_msg = (
                f"Voice separation failed: {e}\n"
                f"Input details:\n"
                f"  - audio_data type: {type(audio_data)}\n"
                f"  - audio_data shape: {getattr(audio_data, 'shape', 'N/A')}\n"
                f"  - audio_data dtype: {getattr(audio_data, 'dtype', 'N/A')}\n"
                f"  - sample_rate: {sample_rate} (type: {type(sample_rate)})\n"
                f"  - expected sample_rate: {self.target_sample_rate}Hz\n"
                f"  - model loaded: {self.model is not None}"
            )
            self.logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        finally:
            self.stop_timer()
    
    async def validate_input(self, audio_data: np.ndarray, sample_rate: int, **kwargs) -> bool:
        """Validate input audio data."""
        if not isinstance(audio_data, np.ndarray):
            self.logger.error("Audio data must be a numpy array")
            return False
        
        if len(audio_data.shape) > 2:
            self.logger.error("Audio data must be 1D or 2D array")
            return False
        
        if sample_rate <= 0:
            self.logger.error("Sample rate must be positive")
            return False
        
        return True
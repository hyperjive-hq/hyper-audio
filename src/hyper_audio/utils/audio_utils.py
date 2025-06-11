"""Audio utility functions for AI Audio Pipeline."""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Tuple, Union, Optional

from .logging_utils import get_logger

logger = get_logger("audio_utils")


def load_audio(
    file_path: Union[str, Path], 
    sample_rate: Optional[int] = None,
    mono: bool = True,
    normalize: bool = True
) -> Tuple[np.ndarray, int]:
    """Load audio file with optional preprocessing.
    
    Args:
        file_path: Path to audio file
        sample_rate: Target sample rate (None to keep original)
        mono: Convert to mono if True
        normalize: Normalize audio amplitude if True
        
    Returns:
        Tuple of (audio_data, sample_rate)
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Audio file not found: {file_path}")
    
    logger.info(f"Loading audio: {file_path}")
    
    try:
        audio, sr = librosa.load(
            str(file_path), 
            sr=sample_rate, 
            mono=mono,
            dtype=np.float32
        )
        
        if normalize:
            audio = librosa.util.normalize(audio)
        
        duration = len(audio) / sr
        logger.info(f"Loaded audio: {duration:.2f}s at {sr}Hz, {'mono' if mono else 'stereo'}")
        
        return audio, sr
        
    except Exception as e:
        logger.error(f"Failed to load audio file {file_path}: {e}")
        raise


def save_audio(
    audio: np.ndarray,
    file_path: Union[str, Path],
    sample_rate: int,
    format: Optional[str] = None
) -> None:
    """Save audio to file.
    
    Args:
        audio: Audio data array
        file_path: Output file path
        sample_rate: Sample rate
        format: Audio format (inferred from extension if None)
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving audio to: {file_path}")
    
    try:
        # Ensure audio is in correct shape for soundfile
        if audio.ndim == 1:
            # Mono audio
            audio_to_save = audio
        elif audio.ndim == 2:
            # Multi-channel audio - transpose for soundfile (samples, channels)
            audio_to_save = audio.T
        else:
            raise ValueError(f"Unsupported audio shape: {audio.shape}")
        
        sf.write(str(file_path), audio_to_save, sample_rate, format=format)
        logger.info(f"Audio saved successfully: {len(audio)/sample_rate:.2f}s")
        
    except Exception as e:
        logger.error(f"Failed to save audio to {file_path}: {e}")
        raise


def resample_audio(
    audio: np.ndarray,
    orig_sr: int,
    target_sr: int
) -> np.ndarray:
    """Resample audio to target sample rate.
    
    Args:
        audio: Audio data
        orig_sr: Original sample rate
        target_sr: Target sample rate
        
    Returns:
        Resampled audio data
    """
    if orig_sr == target_sr:
        return audio
    
    logger.info(f"Resampling audio from {orig_sr}Hz to {target_sr}Hz")
    
    try:
        resampled = librosa.resample(audio, orig_sr=orig_sr, target_sr=target_sr)
        return resampled
    except Exception as e:
        logger.error(f"Failed to resample audio: {e}")
        raise


def ensure_stereo(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is stereo (2-channel).
    
    Args:
        audio: Audio data (mono or stereo)
        
    Returns:
        Stereo audio data with shape (2, samples)
    """
    if audio.ndim == 1:
        # Mono to stereo: duplicate channel
        return np.stack([audio, audio], axis=0)
    elif audio.ndim == 2 and audio.shape[0] == 1:
        # Single channel to stereo
        return np.vstack([audio, audio])
    elif audio.ndim == 2 and audio.shape[0] == 2:
        # Already stereo
        return audio
    else:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    """Ensure audio is mono (1-channel).
    
    Args:
        audio: Audio data (mono or stereo)
        
    Returns:
        Mono audio data
    """
    if audio.ndim == 1:
        return audio
    elif audio.ndim == 2:
        # Convert to mono by averaging channels
        return np.mean(audio, axis=0)
    else:
        raise ValueError(f"Unsupported audio shape: {audio.shape}")


def normalize_audio(audio: np.ndarray, norm: float = 1.0) -> np.ndarray:
    """Normalize audio to specified peak amplitude.
    
    Args:
        audio: Audio data
        norm: Target peak amplitude (default 1.0)
        
    Returns:
        Normalized audio data
    """
    if audio.size == 0:
        return audio
    
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val * norm
    else:
        return audio


def apply_fade(
    audio: np.ndarray,
    fade_in_samples: int = 0,
    fade_out_samples: int = 0
) -> np.ndarray:
    """Apply fade in/out to audio.
    
    Args:
        audio: Audio data
        fade_in_samples: Number of samples for fade in
        fade_out_samples: Number of samples for fade out
        
    Returns:
        Audio with fades applied
    """
    audio_faded = audio.copy()
    
    if fade_in_samples > 0:
        fade_in = np.linspace(0, 1, fade_in_samples)
        if audio.ndim == 1:
            audio_faded[:fade_in_samples] *= fade_in
        else:
            audio_faded[:, :fade_in_samples] *= fade_in
    
    if fade_out_samples > 0:
        fade_out = np.linspace(1, 0, fade_out_samples)
        if audio.ndim == 1:
            audio_faded[-fade_out_samples:] *= fade_out
        else:
            audio_faded[:, -fade_out_samples:] *= fade_out
    
    return audio_faded
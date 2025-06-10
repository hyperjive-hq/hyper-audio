"""Configuration settings for Hyper Audio."""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Configuration settings for the Hyper Audio Pipeline."""
    
    def __init__(self):
        # Load settings from environment variables with defaults
        self.huggingface_token = os.getenv("HUGGINGFACE_TOKEN", "")
        self.torch_device = os.getenv("TORCH_DEVICE", "cuda")
        self.cuda_visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "0")
        
        # Paths
        self.model_cache_dir = Path(os.getenv("MODEL_CACHE_DIR", "~/.cache/hyper_audio")).expanduser()
        self.log_file = Path(os.getenv("LOG_FILE", "logs/hyper_audio.log"))
        
        # Audio settings
        self.default_sample_rate = int(os.getenv("DEFAULT_SAMPLE_RATE", "16000"))
        self.max_audio_length = int(os.getenv("MAX_AUDIO_LENGTH", "3600"))
        
        # Model configurations
        self.whisper_model = os.getenv("WHISPER_MODEL", "large-v2")
        self.demucs_model = os.getenv("DEMUCS_MODEL", "htdemucs")
        self.diarization_model = os.getenv("DIARIZATION_MODEL", "pyannote/speaker-diarization")
        self.tts_model = os.getenv("TTS_MODEL", "metavoiceio/metavoice-1B-v0.1")
        
        # Performance settings
        self.use_fp16 = os.getenv("USE_FP16", "true").lower() == "true"
        self.batch_size = int(os.getenv("BATCH_SIZE", "8"))
        self.max_workers = int(os.getenv("MAX_WORKERS", "4"))
        
        # Logging
        self.log_level = os.getenv("LOG_LEVEL", "INFO")
        
        # Create directories if they don't exist
        self.model_cache_dir.mkdir(parents=True, exist_ok=True)
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
    
    @property
    def device(self) -> str:
        """Get the torch device, checking CUDA availability."""
        if self.torch_device == "cuda":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.torch_device
    
    def validate(self) -> bool:
        """Validate configuration settings."""
        if not self.huggingface_token:
            print("Warning: HUGGINGFACE_TOKEN not set. Some models may not be accessible.")
            return False
        return True
    
    def __repr__(self) -> str:
        """String representation of settings."""
        return f"Settings(device={self.device}, sample_rate={self.default_sample_rate}, fp16={self.use_fp16})"


# Global settings instance
settings = Settings()
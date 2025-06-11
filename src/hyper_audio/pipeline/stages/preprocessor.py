"""Audio preprocessing stage."""

import numpy as np
from typing import Tuple, Union, Optional
from pathlib import Path

from .base import BasePipelineStage
from ...utils.audio_utils import load_audio, normalize_audio


class AudioPreprocessor(BasePipelineStage):
    """Audio preprocessing and normalization stage."""
    
    def __init__(self):
        super().__init__("AudioPreprocessor")
    
    async def process(self, input_path: Union[str, Path], **kwargs) -> Tuple[np.ndarray, int]:
        """Process audio file through preprocessing.
        
        Args:
            input_path: Path to input audio file
            
        Returns:
            Tuple of (processed_audio, sample_rate)
        """
        self.start_timer()
        
        try:
            self.logger.info(f"Processing audio file: {input_path}")
            
            # Load audio
            audio_data, sample_rate = load_audio(str(input_path))
            
            # Normalize audio
            normalized_audio = normalize_audio(audio_data)
            
            self.logger.info(f"Audio preprocessing completed. Shape: {normalized_audio.shape}, SR: {sample_rate}")
            
            return normalized_audio, sample_rate
            
        except Exception as e:
            self.logger.error(f"Audio preprocessing failed: {e}")
            raise
        finally:
            self.stop_timer()
    
    async def validate_input(self, input_path: Union[str, Path], **kwargs) -> bool:
        """Validate input audio file."""
        path = Path(input_path)
        if not path.exists():
            self.logger.error(f"Input file does not exist: {input_path}")
            return False
        
        if not path.is_file():
            self.logger.error(f"Input path is not a file: {input_path}")
            return False
        
        return True
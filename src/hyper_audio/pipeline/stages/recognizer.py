"""Speech recognition stage."""

import numpy as np
from typing import Dict, List, Any

from .base import BasePipelineStage


class SpeechRecognizer(BasePipelineStage):
    """Speech transcription stage."""
    
    def __init__(self):
        super().__init__("SpeechRecognizer")
    
    async def process(self, vocals_audio: np.ndarray, sample_rate: int, 
                     speaker_segments: List[Dict[str, Any]], **kwargs) -> Dict[str, Any]:
        """Transcribe speech from vocal audio with speaker alignment.
        
        Args:
            vocals_audio: Separated vocal audio
            sample_rate: Audio sample rate
            speaker_segments: Speaker diarization results
            
        Returns:
            Dictionary with transcription text and aligned segments
        """
        self.start_timer()
        
        try:
            self.logger.info(f"Transcribing speech. Audio shape: {vocals_audio.shape}")
            
            # Placeholder implementation - in practice this would use 
            # models like Whisper, Wav2Vec2, or similar
            
            # Create mock transcription aligned with speaker segments
            segments = []
            full_text_parts = []
            
            # Mock phrases to cycle through
            mock_phrases = [
                "Hello, how are you doing today?",
                "I'm doing well, thank you for asking.",
                "That's great to hear.",
                "What are your plans for the weekend?",
                "I'm thinking of going hiking.",
                "That sounds like a wonderful idea."
            ]
            
            for i, speaker_segment in enumerate(speaker_segments):
                # Use a different phrase for each segment
                phrase = mock_phrases[i % len(mock_phrases)]
                
                segment = {
                    "text": phrase,
                    "start": speaker_segment["start"],
                    "end": speaker_segment["end"],
                    "speaker": speaker_segment["speaker"],
                    "confidence": 0.92  # Mock confidence
                }
                
                segments.append(segment)
                full_text_parts.append(f"[{speaker_segment['speaker']}] {phrase}")
            
            result = {
                "text": " ".join(full_text_parts),
                "segments": segments,
                "language": "en",  # Mock language detection
                "total_segments": len(segments)
            }
            
            self.logger.info(f"Speech recognition completed. {len(segments)} segments transcribed")
            return result
            
        except Exception as e:
            self.logger.error(f"Speech recognition failed: {e}")
            raise
        finally:
            self.stop_timer()
    
    async def validate_input(self, vocals_audio: np.ndarray, sample_rate: int, 
                           speaker_segments: List[Dict[str, Any]], **kwargs) -> bool:
        """Validate input parameters."""
        if not isinstance(vocals_audio, np.ndarray):
            self.logger.error("Vocals audio must be a numpy array")
            return False
        
        if len(vocals_audio) == 0:
            self.logger.error("Vocals audio cannot be empty")
            return False
        
        if not isinstance(speaker_segments, list):
            self.logger.error("Speaker segments must be a list")
            return False
        
        if len(speaker_segments) == 0:
            self.logger.error("Speaker segments cannot be empty")
            return False
        
        # Validate segment structure
        for segment in speaker_segments:
            if not isinstance(segment, dict):
                self.logger.error("Each speaker segment must be a dictionary")
                return False
            
            required_keys = ["speaker", "start", "end"]
            if not all(key in segment for key in required_keys):
                self.logger.error(f"Speaker segment missing required keys: {required_keys}")
                return False
        
        return True
"""Audio reconstruction stage."""

import numpy as np
from typing import Dict, List, Any, Optional

from .base import BasePipelineStage


class AudioReconstructor(BasePipelineStage):
    """Audio reconstruction stage - combines synthesized voice with original music."""
    
    def __init__(self):
        super().__init__("AudioReconstructor")
    
    async def process(self, separated_audio: Dict[str, np.ndarray], 
                     synthesized_audio: Dict[str, np.ndarray],
                     transcription: Dict[str, Any],
                     sample_rate: int,
                     target_speaker: Optional[str] = None, **kwargs) -> np.ndarray:
        """Reconstruct final audio by combining synthesized voice with original music.
        
        Args:
            separated_audio: Original separated vocals and music
            synthesized_audio: New synthesized voice for target speaker
            transcription: Transcription with timing information
            sample_rate: Audio sample rate
            target_speaker: Speaker that was replaced
            
        Returns:
            Final reconstructed audio
        """
        self.start_timer()
        
        try:
            self.logger.info("Reconstructing final audio")
            
            # Get original components
            original_vocals = separated_audio.get("vocals", np.array([]))
            original_music = separated_audio.get("music", np.array([]))
            
            # Determine target speaker if not specified
            segments = transcription.get("segments", [])
            if target_speaker is None and segments:
                target_speaker = segments[0]["speaker"]
            
            # Start with original vocals as base
            reconstructed_vocals = original_vocals.copy() if len(original_vocals) > 0 else np.array([])
            
            # Replace target speaker segments with synthesized voice
            if target_speaker in synthesized_audio:
                synthesized = synthesized_audio[target_speaker]
                
                # For simplicity, replace entire vocal track if we have synthesized audio
                # In practice, this would involve precise timing alignment
                if len(synthesized) > 0:
                    # Pad or trim synthesized audio to match original length
                    if len(reconstructed_vocals) > 0:
                        target_length = len(reconstructed_vocals)
                        if len(synthesized) > target_length:
                            synthesized = synthesized[:target_length]
                        elif len(synthesized) < target_length:
                            # Pad with silence
                            padding = np.zeros(target_length - len(synthesized))
                            synthesized = np.concatenate([synthesized, padding])
                        
                        # Replace original vocals with synthesized (simplified approach)
                        reconstructed_vocals = synthesized
                        self.logger.info(f"Replaced vocals with synthesized voice for {target_speaker}")
                    else:
                        reconstructed_vocals = synthesized
                        self.logger.info(f"Used synthesized voice as primary vocal track")
            
            # Combine vocals and music
            if len(original_music) > 0 and len(reconstructed_vocals) > 0:
                # Ensure both have the same length
                min_length = min(len(original_music), len(reconstructed_vocals))
                final_audio = (original_music[:min_length] * 0.7 + 
                             reconstructed_vocals[:min_length] * 0.8)  # Mix with levels
            elif len(reconstructed_vocals) > 0:
                final_audio = reconstructed_vocals
            elif len(original_music) > 0:
                final_audio = original_music
            else:
                # Fallback - create silence
                final_audio = np.zeros(sample_rate)  # 1 second of silence
                self.logger.warning("No audio components available, created silence")
            
            # Normalize to prevent clipping
            if len(final_audio) > 0:
                max_val = np.max(np.abs(final_audio))
                if max_val > 0:
                    final_audio = final_audio / max_val * 0.95  # Normalize to 95% to prevent clipping
            
            self.logger.info(f"Audio reconstruction completed. Final length: {len(final_audio)/sample_rate:.2f}s")
            return final_audio
            
        except Exception as e:
            self.logger.error(f"Audio reconstruction failed: {e}")
            raise
        finally:
            self.stop_timer()
    
    async def validate_input(self, separated_audio: Dict[str, np.ndarray], 
                           synthesized_audio: Dict[str, np.ndarray],
                           transcription: Dict[str, Any],
                           sample_rate: int, **kwargs) -> bool:
        """Validate input parameters."""
        if not isinstance(separated_audio, dict):
            self.logger.error("Separated audio must be a dictionary")
            return False
        
        if not isinstance(synthesized_audio, dict):
            self.logger.error("Synthesized audio must be a dictionary")
            return False
        
        if not isinstance(transcription, dict):
            self.logger.error("Transcription must be a dictionary")
            return False
        
        # Check that we have at least some audio data
        has_vocals = "vocals" in separated_audio and len(separated_audio["vocals"]) > 0
        has_music = "music" in separated_audio and len(separated_audio["music"]) > 0
        has_synthesized = any(len(audio) > 0 for audio in synthesized_audio.values())
        
        if not (has_vocals or has_music or has_synthesized):
            self.logger.error("No valid audio data provided for reconstruction")
            return False
        
        if sample_rate <= 0:
            self.logger.error("Sample rate must be positive")
            return False
        
        return True
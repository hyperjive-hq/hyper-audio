"""Audio reconstruction stage."""

import numpy as np
from typing import Dict, Any, Optional

from ..stage_interface import (
    EnhancedPipelineStage, StageMetadata, StageInput, StageOutput,
    DataType
)


class AudioReconstructor(EnhancedPipelineStage):
    """Audio reconstruction stage - combines synthesized voice with original music."""

    def __init__(self, stage_id: str = None, config: Dict[str, Any] = None):
        super().__init__(stage_id, config)
        self.vocal_level = self.config.get("vocal_level", 0.8)
        self.music_level = self.config.get("music_level", 0.7)
        self.target_speaker = self.config.get("target_speaker", None)

    def get_metadata(self) -> StageMetadata:
        """Define the stage's inputs, outputs, and capabilities."""
        return StageMetadata(
            name="Audio Reconstructor",
            description="Combines synthesized voice with original music to create final audio",
            category="reconstruction",
            performance_notes=f"Vocal level: {self.vocal_level}, Music level: {self.music_level}",
            inputs=[
                StageInput(
                    name="separated_audio",
                    data_type=DataType.SEPARATED_AUDIO,
                    required=True,
                    description="Original separated vocals and music"
                ),
                StageInput(
                    name="synthesized_audio",
                    data_type=DataType.SYNTHESIZED_AUDIO,
                    required=True,
                    description="New synthesized voice for target speaker"
                ),
                StageInput(
                    name="transcription",
                    data_type=DataType.TRANSCRIPTION,
                    required=True,
                    description="Transcription with timing information"
                ),
                StageInput(
                    name="sample_rate",
                    data_type=DataType.SAMPLE_RATE,
                    required=True,
                    description="Audio sample rate"
                )
            ],
            outputs=[
                StageOutput(
                    name="final_audio",
                    data_type=DataType.FINAL_AUDIO,
                    description="Final reconstructed audio with synthesized voice"
                )
            ]
        )

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Reconstruct final audio by combining synthesized voice with original music.
        
        Args:
            inputs: Dictionary containing required audio components
            
        Returns:
            Dictionary with final reconstructed audio
        """
        await self.validate_inputs(inputs)
        
        separated_audio = inputs["separated_audio"]
        synthesized_audio = inputs["synthesized_audio"]
        transcription = inputs["transcription"]
        sample_rate = inputs["sample_rate"]

        try:
            # Get original components
            original_vocals = separated_audio.get("vocals", np.array([]))
            original_music = separated_audio.get("music", np.array([]))

            # Determine target speaker if not specified
            target_speaker = self.target_speaker
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
                    else:
                        reconstructed_vocals = synthesized

            # Combine vocals and music
            if len(original_music) > 0 and len(reconstructed_vocals) > 0:
                # Ensure both have the same length
                min_length = min(len(original_music), len(reconstructed_vocals))
                final_audio = (original_music[:min_length] * self.music_level +
                             reconstructed_vocals[:min_length] * self.vocal_level)  # Mix with levels
            elif len(reconstructed_vocals) > 0:
                final_audio = reconstructed_vocals
            elif len(original_music) > 0:
                final_audio = original_music
            else:
                # Fallback - create silence
                final_audio = np.zeros(sample_rate)  # 1 second of silence

            # Normalize to prevent clipping
            if len(final_audio) > 0:
                max_val = np.max(np.abs(final_audio))
                if max_val > 0:
                    final_audio = final_audio / max_val * 0.95  # Normalize to 95% to prevent clipping

            return {
                "final_audio": final_audio
            }

        except Exception as e:
            raise RuntimeError(f"Audio reconstruction failed: {e}") from e

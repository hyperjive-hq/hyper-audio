"""Speech recognition stage."""

import numpy as np
from typing import Dict, List, Any

from ..stage_interface import (
    EnhancedPipelineStage, StageMetadata, StageInput, StageOutput,
    DataType
)


class SpeechRecognizer(EnhancedPipelineStage):
    """Speech transcription stage."""

    def __init__(self, stage_id: str = None, config: Dict[str, Any] = None):
        super().__init__(stage_id, config)
        self.model_name = self.config.get("model_name", "whisper-base")
        self.language = self.config.get("language", "auto")

    def get_metadata(self) -> StageMetadata:
        """Define the stage's inputs, outputs, and capabilities."""
        return StageMetadata(
            name="Speech Recognizer",
            description="Transcribes speech from vocal audio with speaker alignment",
            category="recognition",
            model_name=self.model_name,
            performance_notes=f"Language: {self.language}",
            inputs=[
                StageInput(
                    name="vocals",
                    data_type=DataType.AUDIO_MONO,
                    required=True,
                    description="Separated vocal audio"
                ),
                StageInput(
                    name="sample_rate",
                    data_type=DataType.SAMPLE_RATE,
                    required=True,
                    description="Audio sample rate"
                ),
                StageInput(
                    name="speaker_segments",
                    data_type=DataType.SPEAKER_SEGMENTS,
                    required=True,
                    description="Speaker diarization results"
                )
            ],
            outputs=[
                StageOutput(
                    name="transcription",
                    data_type=DataType.TRANSCRIPTION,
                    description="Transcription with text, segments, and metadata"
                )
            ]
        )

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Transcribe speech from vocal audio with speaker alignment.
        
        Args:
            inputs: Dictionary containing 'vocals', 'sample_rate', and 'speaker_segments' keys
            
        Returns:
            Dictionary with transcription results
        """
        await self.validate_inputs(inputs)
        
        vocals_audio = inputs["vocals"]
        sample_rate = inputs["sample_rate"]
        speaker_segments = inputs["speaker_segments"]

        try:
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

            transcription = {
                "full_text": " ".join(full_text_parts),
                "segments": segments,
                "language": "en",  # Mock language detection
                "total_segments": len(segments)
            }

            return {
                "transcription": transcription
            }

        except Exception as e:
            raise RuntimeError(f"Speech recognition failed: {e}") from e

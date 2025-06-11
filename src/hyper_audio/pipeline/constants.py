"""Constants for the audio processing pipeline."""

from enum import Enum

# Stage configuration
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_SECONDS = 5
DEFAULT_CHECKPOINT_DIR = "checkpoints"

# Performance thresholds
SLOW_STAGE_THRESHOLD_SECONDS = 300  # 5 minutes
HIGH_MEMORY_THRESHOLD_GB = 20
HIGH_RETRY_THRESHOLD = 5

# File extensions and formats
CHECKPOINT_EXTENSION = ".pkl"
STATE_FILENAME = "pipeline_state.json"
FAILURE_REPORT_FILENAME = "failure_report.json"
CHECKSUM_EXTENSION = ".txt"

# Analytics
DEFAULT_ANALYSIS_DAYS = 30
RECENT_JOBS_LIMIT = 10


class StageStatus(Enum):
    """Pipeline stage execution status."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class JobStatus(Enum):
    """Overall job status."""
    NOT_FOUND = "not_found"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    ERROR = "error"


# Stage definitions - centralized configuration
PIPELINE_STAGES = [
    {"name": "preprocessor", "class": "AudioPreprocessor", "description": "Audio preprocessing and normalization"},
    {"name": "separator", "class": "VoiceSeparator", "description": "Voice/music separation"}, 
    {"name": "diarizer", "class": "SpeakerDiarizer", "description": "Speaker identification"},
    {"name": "recognizer", "class": "SpeechRecognizer", "description": "Speech transcription"},
    {"name": "synthesizer", "class": "VoiceSynthesizer", "description": "Voice synthesis"},
    {"name": "reconstructor", "class": "AudioReconstructor", "description": "Audio reconstruction"}
]
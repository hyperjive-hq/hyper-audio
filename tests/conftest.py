"""Pytest configuration and shared fixtures."""

import pytest
import tempfile
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from unittest.mock import Mock, AsyncMock, patch

from src.hyper_audio.pipeline.models import PipelineState, StageMetrics
from src.hyper_audio.pipeline.constants import StageStatus, JobStatus
from src.hyper_audio.pipeline.checkpoint import CheckpointManager, StateManager


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def checkpoint_dir(temp_dir):
    """Create a checkpoint directory for tests."""
    checkpoint_path = temp_dir / "checkpoints"
    checkpoint_path.mkdir(parents=True, exist_ok=True)
    return checkpoint_path


@pytest.fixture
def job_id():
    """Standard job ID for testing."""
    return "test_job_20231201_120000"


@pytest.fixture
def sample_audio_data():
    """Generate sample audio data for testing."""
    # Generate 1 second of sine wave at 16kHz
    sample_rate = 16000
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    frequency = 440  # A4 note
    audio = np.sin(2 * np.pi * frequency * t).astype(np.float32)
    return audio, sample_rate


@pytest.fixture
def sample_audio_file(temp_dir, sample_audio_data):
    """Create a sample audio file for testing."""
    import soundfile as sf
    
    audio, sample_rate = sample_audio_data
    audio_file = temp_dir / "test_audio.wav"
    sf.write(str(audio_file), audio, sample_rate)
    return audio_file


@pytest.fixture
def sample_stage_metrics():
    """Create sample stage metrics for testing."""
    return [
        StageMetrics(
            stage_name="preprocessor",
            status=StageStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration_seconds=5.2,
            memory_peak_gb=2.1,
            retry_count=0,
            checkpoint_size_mb=15.3
        ),
        StageMetrics(
            stage_name="separator",
            status=StageStatus.COMPLETED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration_seconds=12.8,
            memory_peak_gb=8.5,
            retry_count=1,
            checkpoint_size_mb=45.7
        ),
        StageMetrics(
            stage_name="diarizer",
            status=StageStatus.FAILED,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            duration_seconds=3.1,
            error_message="CUDA out of memory",
            error_type="RuntimeError",
            retry_count=3
        )
    ]


@pytest.fixture
def sample_pipeline_state(job_id, sample_stage_metrics):
    """Create a sample pipeline state for testing."""
    return PipelineState(
        job_id=job_id,
        input_path="/test/input.wav",
        output_path="/test/output.wav",
        config={"max_retries": 3, "use_fp16": True},
        current_stage=2,
        stages_completed=["preprocessor", "separator"],
        stage_metrics=sample_stage_metrics,
        data_checksums={"preprocessor": "abc123", "separator": "def456"}
    )


@pytest.fixture
def checkpoint_manager(checkpoint_dir, job_id):
    """Create a checkpoint manager for testing."""
    job_checkpoint_dir = checkpoint_dir / job_id
    return CheckpointManager(job_checkpoint_dir)


@pytest.fixture
def state_manager(checkpoint_dir, job_id):
    """Create a state manager for testing."""
    job_checkpoint_dir = checkpoint_dir / job_id
    return StateManager(job_checkpoint_dir)


@pytest.fixture
def mock_torch():
    """Mock PyTorch CUDA functionality."""
    with patch('torch.cuda.is_available', return_value=True), \
         patch('torch.cuda.memory_allocated', return_value=1024**3), \
         patch('torch.cuda.memory_reserved', return_value=2*1024**3), \
         patch('torch.cuda.max_memory_allocated', return_value=3*1024**3), \
         patch('torch.cuda.empty_cache') as mock_empty_cache, \
         patch('torch.cuda.reset_peak_memory_stats') as mock_reset_stats:
        
        yield {
            'empty_cache': mock_empty_cache,
            'reset_stats': mock_reset_stats
        }


@pytest.fixture
def mock_pipeline_stages():
    """Create mock pipeline stages for testing."""
    stages = {}
    
    for stage_name in ["preprocessor", "separator", "diarizer", "recognizer", "synthesizer", "reconstructor"]:
        mock_stage = AsyncMock()
        mock_stage.process = AsyncMock()
        mock_stage.cleanup = AsyncMock()
        stages[stage_name] = mock_stage
    
    return stages


@pytest.fixture
def mock_audio_utils():
    """Mock audio utility functions."""
    with patch('src.hyper_audio.utils.audio_utils.load_audio') as mock_load, \
         patch('src.hyper_audio.utils.audio_utils.save_audio') as mock_save:
        
        # Configure mock load_audio to return sample data
        mock_load.return_value = (np.random.rand(16000).astype(np.float32), 16000)
        
        yield {
            'load_audio': mock_load,
            'save_audio': mock_save
        }


@pytest.fixture
def mock_progress_callback():
    """Create a mock progress callback for testing."""
    return Mock()


class AsyncContextManager:
    """Helper for creating async context managers in tests."""
    
    def __init__(self, return_value=None):
        self.return_value = return_value
    
    async def __aenter__(self):
        return self.return_value
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        pass


@pytest.fixture
def async_context_manager():
    """Factory for creating async context managers."""
    return AsyncContextManager


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "gpu: mark test as requiring GPU/CUDA"
    )
    config.addinivalue_line(
        "markers", "nvidia: mark test as requiring NVIDIA GPU"
    )


def has_cuda_support():
    """Check if CUDA is available for testing."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def has_nvidia_gpu():
    """Check if NVIDIA GPU is available for testing."""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


# Skip GPU tests if CUDA is not available
def pytest_collection_modifyitems(config, items):
    """Modify test collection to handle GPU tests."""
    skip_gpu = not has_cuda_support()
    skip_nvidia = not has_nvidia_gpu()
    
    if skip_gpu:
        gpu_marker = pytest.mark.skip(reason="GPU/CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(gpu_marker)
    
    if skip_nvidia:
        nvidia_marker = pytest.mark.skip(reason="NVIDIA GPU not available")
        for item in items:
            if "nvidia" in item.keywords:
                item.add_marker(nvidia_marker)
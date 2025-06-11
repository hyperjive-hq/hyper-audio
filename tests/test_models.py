"""Tests for pipeline models and data structures."""

import pytest
import numpy as np
from datetime import datetime, timezone
from unittest.mock import Mock, patch

from src.hyper_audio.pipeline.models import (
    StageMetrics, PipelineState, JobSummary, PipelineResult
)
from src.hyper_audio.pipeline.constants import StageStatus, JobStatus


class TestStageMetrics:
    """Test StageMetrics data model."""
    
    def test_creation(self):
        """Test creating StageMetrics."""
        metrics = StageMetrics(
            stage_name="test_stage",
            status=StageStatus.COMPLETED,
            duration_seconds=10.5,
            memory_peak_gb=2.3
        )
        
        assert metrics.stage_name == "test_stage"
        assert metrics.status == StageStatus.COMPLETED
        assert metrics.duration_seconds == 10.5
        assert metrics.memory_peak_gb == 2.3
        assert metrics.retry_count == 0  # default value
    
    def test_to_dict(self):
        """Test converting StageMetrics to dictionary."""
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)
        
        metrics = StageMetrics(
            stage_name="test_stage",
            status=StageStatus.FAILED,
            start_time=start_time,
            end_time=end_time,
            error_message="Test error",
            error_type="TestException",
            retry_count=2
        )
        
        data = metrics.to_dict()
        
        assert data["stage_name"] == "test_stage"
        assert data["status"] == "failed"  # enum value
        assert data["start_time"] == start_time.isoformat()
        assert data["end_time"] == end_time.isoformat()
        assert data["error_message"] == "Test error"
        assert data["error_type"] == "TestException"
        assert data["retry_count"] == 2
    
    def test_from_dict(self):
        """Test creating StageMetrics from dictionary."""
        start_time = datetime.now(timezone.utc)
        end_time = datetime.now(timezone.utc)
        
        data = {
            "stage_name": "test_stage",
            "status": "completed",
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": 15.3,
            "memory_peak_gb": 4.2,
            "retry_count": 1
        }
        
        metrics = StageMetrics.from_dict(data)
        
        assert metrics.stage_name == "test_stage"
        assert metrics.status == StageStatus.COMPLETED
        assert metrics.start_time == start_time
        assert metrics.end_time == end_time
        assert metrics.duration_seconds == 15.3
        assert metrics.memory_peak_gb == 4.2
        assert metrics.retry_count == 1
    
    def test_from_dict_with_none_times(self):
        """Test creating StageMetrics from dict with None datetime fields."""
        data = {
            "stage_name": "test_stage",
            "status": "pending",
            "start_time": None,
            "end_time": None,
            "retry_count": 0
        }
        
        metrics = StageMetrics.from_dict(data)
        
        assert metrics.stage_name == "test_stage"
        assert metrics.status == StageStatus.PENDING
        assert metrics.start_time is None
        assert metrics.end_time is None


class TestPipelineState:
    """Test PipelineState data model."""
    
    def test_creation(self, job_id, sample_stage_metrics):
        """Test creating PipelineState."""
        state = PipelineState(
            job_id=job_id,
            input_path="/test/input.wav",
            output_path="/test/output.wav",
            config={"max_retries": 3},
            current_stage=2,
            stages_completed=["stage1", "stage2"],
            stage_metrics=sample_stage_metrics
        )
        
        assert state.job_id == job_id
        assert state.input_path == "/test/input.wav"
        assert state.output_path == "/test/output.wav"
        assert state.config["max_retries"] == 3
        assert state.current_stage == 2
        assert len(state.stages_completed) == 2
        assert len(state.stage_metrics) == len(sample_stage_metrics)
    
    def test_default_values(self, job_id):
        """Test PipelineState default values."""
        state = PipelineState(
            job_id=job_id,
            input_path="/test/input.wav",
            output_path="/test/output.wav",
            config={}
        )
        
        assert state.current_stage == 0
        assert state.stages_completed == []
        assert state.stage_metrics == []
        assert state.data_checksums == {}
        assert isinstance(state.created_at, datetime)
        assert isinstance(state.updated_at, datetime)
    
    def test_to_dict(self, sample_pipeline_state):
        """Test converting PipelineState to dictionary."""
        data = sample_pipeline_state.to_dict()
        
        assert data["job_id"] == sample_pipeline_state.job_id
        assert data["input_path"] == sample_pipeline_state.input_path
        assert data["output_path"] == sample_pipeline_state.output_path
        assert data["current_stage"] == sample_pipeline_state.current_stage
        assert isinstance(data["created_at"], str)  # serialized
        assert isinstance(data["updated_at"], str)  # serialized
        assert isinstance(data["stage_metrics"], list)
        assert len(data["stage_metrics"]) == len(sample_pipeline_state.stage_metrics)
    
    def test_from_dict(self, sample_pipeline_state):
        """Test creating PipelineState from dictionary."""
        # Convert to dict and back
        data = sample_pipeline_state.to_dict()
        recreated_state = PipelineState.from_dict(data)
        
        assert recreated_state.job_id == sample_pipeline_state.job_id
        assert recreated_state.input_path == sample_pipeline_state.input_path
        assert recreated_state.output_path == sample_pipeline_state.output_path
        assert recreated_state.current_stage == sample_pipeline_state.current_stage
        assert recreated_state.created_at == sample_pipeline_state.created_at
        assert recreated_state.updated_at == sample_pipeline_state.updated_at
        assert len(recreated_state.stage_metrics) == len(sample_pipeline_state.stage_metrics)
    
    def test_update_timestamp(self, sample_pipeline_state):
        """Test updating timestamp."""
        original_time = sample_pipeline_state.updated_at
        
        # Wait a bit and update
        import time
        time.sleep(0.01)
        sample_pipeline_state.update_timestamp()
        
        assert sample_pipeline_state.updated_at > original_time
    
    def test_add_stage_metrics(self, sample_pipeline_state):
        """Test adding stage metrics."""
        original_count = len(sample_pipeline_state.stage_metrics)
        original_time = sample_pipeline_state.updated_at
        
        new_metrics = StageMetrics(
            stage_name="new_stage",
            status=StageStatus.COMPLETED
        )
        
        import time
        time.sleep(0.01)
        sample_pipeline_state.add_stage_metrics(new_metrics)
        
        assert len(sample_pipeline_state.stage_metrics) == original_count + 1
        assert sample_pipeline_state.stage_metrics[-1] == new_metrics
        assert sample_pipeline_state.updated_at > original_time
    
    def test_mark_stage_completed(self, sample_pipeline_state):
        """Test marking stage as completed."""
        original_stage = sample_pipeline_state.current_stage
        original_completed = len(sample_pipeline_state.stages_completed)
        
        sample_pipeline_state.mark_stage_completed("new_stage")
        
        assert sample_pipeline_state.current_stage == original_stage + 1
        assert len(sample_pipeline_state.stages_completed) == original_completed + 1
        assert "new_stage" in sample_pipeline_state.stages_completed
    
    def test_get_stage_status(self, sample_pipeline_state):
        """Test getting stage status."""
        # Test existing stage
        assert sample_pipeline_state.get_stage_status("preprocessor") == StageStatus.COMPLETED
        assert sample_pipeline_state.get_stage_status("diarizer") == StageStatus.FAILED
        
        # Test non-existent stage
        assert sample_pipeline_state.get_stage_status("nonexistent") == StageStatus.PENDING
    
    def test_get_overall_status(self, sample_pipeline_state):
        """Test getting overall job status."""
        # Test with failed stage
        assert sample_pipeline_state.get_overall_status(6) == JobStatus.FAILED
        
        # Test in progress
        state_in_progress = PipelineState(
            job_id="test",
            input_path="/test/input.wav",
            output_path="/test/output.wav",
            config={},
            current_stage=2,
            stage_metrics=[
                StageMetrics(stage_name="stage1", status=StageStatus.COMPLETED),
                StageMetrics(stage_name="stage2", status=StageStatus.COMPLETED)
            ]
        )
        assert state_in_progress.get_overall_status(6) == JobStatus.IN_PROGRESS
        
        # Test completed
        state_completed = PipelineState(
            job_id="test",
            input_path="/test/input.wav",
            output_path="/test/output.wav",
            config={},
            current_stage=6
        )
        assert state_completed.get_overall_status(6) == JobStatus.COMPLETED


class TestJobSummary:
    """Test JobSummary data model."""
    
    def test_from_state_completed(self, sample_pipeline_state):
        """Test creating JobSummary from completed state."""
        # Modify state to be completed without errors
        sample_pipeline_state.current_stage = 6
        sample_pipeline_state.stages_completed = ["stage1", "stage2", "stage3", "stage4", "stage5", "stage6"]
        
        # Remove the failed stage metrics and replace with completed ones
        from src.hyper_audio.pipeline.constants import StageStatus
        from datetime import datetime, timezone
        
        completed_metrics = [
            StageMetrics(
                stage_name=f"stage{i}",
                status=StageStatus.COMPLETED,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                duration_seconds=5.0,
                memory_peak_gb=2.0,
                retry_count=0
            ) for i in range(1, 7)
        ]
        sample_pipeline_state.stage_metrics = completed_metrics
        
        summary = JobSummary.from_state(sample_pipeline_state, 6)
        
        assert summary.job_id == sample_pipeline_state.job_id
        assert summary.status == JobStatus.COMPLETED
        assert summary.progress_percentage == 100.0
        assert summary.current_stage == 6
        assert summary.total_stages == 6
        assert len(summary.stages_completed) == 6
        assert summary.error_message is None
    
    def test_from_state_failed(self, sample_pipeline_state):
        """Test creating JobSummary from failed state."""
        summary = JobSummary.from_state(sample_pipeline_state, 6)
        
        assert summary.job_id == sample_pipeline_state.job_id
        assert summary.status == JobStatus.FAILED
        assert summary.progress_percentage == (2 / 6) * 100  # 2 completed out of 6
        assert summary.current_stage == 2
        assert summary.total_stages == 6
        assert summary.error_message == "CUDA out of memory"  # from failed stage
    
    def test_from_state_in_progress(self):
        """Test creating JobSummary from in-progress state."""
        state = PipelineState(
            job_id="test_job",
            input_path="/test/input.wav",
            output_path="/test/output.wav",
            config={},
            current_stage=3,
            stages_completed=["stage1", "stage2", "stage3"],
            stage_metrics=[
                StageMetrics(stage_name="stage1", status=StageStatus.COMPLETED),
                StageMetrics(stage_name="stage2", status=StageStatus.COMPLETED),
                StageMetrics(stage_name="stage3", status=StageStatus.COMPLETED)
            ]
        )
        
        summary = JobSummary.from_state(state, 6)
        
        assert summary.status == JobStatus.IN_PROGRESS
        assert summary.progress_percentage == 50.0  # 3 out of 6
        assert summary.error_message is None


class TestPipelineResult:
    """Test PipelineResult data model."""
    
    def test_creation(self, job_id, checkpoint_manager):
        """Test creating PipelineResult."""
        result = PipelineResult(job_id, checkpoint_manager)
        
        assert result.job_id == job_id
        assert result.checkpoint_manager == checkpoint_manager
        assert result.processing_stats == {}
        assert result.stage_outputs == {}
    
    def test_save_stage_data(self, job_id, checkpoint_manager, sample_audio_data):
        """Test saving stage data through result."""
        result = PipelineResult(job_id, checkpoint_manager)
        
        # Mock the checkpoint manager's save method
        mock_path = Mock()
        checkpoint_manager.save_stage_data = Mock(return_value=mock_path)
        
        path = result.save_stage_data("preprocessor", sample_audio_data)
        
        assert path == mock_path
        assert result.stage_outputs["preprocessor"] == mock_path
        checkpoint_manager.save_stage_data.assert_called_once_with("preprocessor", sample_audio_data)
    
    def test_get_stage_data_preprocessor(self, job_id, checkpoint_manager, sample_audio_data):
        """Test getting preprocessor stage data."""
        result = PipelineResult(job_id, checkpoint_manager)
        
        # Mock the checkpoint manager's load method
        checkpoint_manager.load_stage_data = Mock(return_value=sample_audio_data)
        
        audio, sample_rate = result.get_stage_data("preprocessor")
        
        assert audio is not None
        assert sample_rate is not None
        checkpoint_manager.load_stage_data.assert_called_once_with("preprocessor")
    
    def test_get_stage_data_separator(self, job_id, checkpoint_manager):
        """Test getting separator stage data."""
        result = PipelineResult(job_id, checkpoint_manager)
        
        separated_audio = {"vocals": np.array([1, 2, 3]), "music": np.array([4, 5, 6])}
        checkpoint_manager.load_stage_data = Mock(return_value=separated_audio)
        
        data = result.get_stage_data("separator")
        
        assert "vocals" in data
        assert "music" in data
        checkpoint_manager.load_stage_data.assert_called_once_with("separator")
    
    def test_get_stage_data_invalid_stage(self, job_id, checkpoint_manager):
        """Test getting data for invalid stage raises error."""
        result = PipelineResult(job_id, checkpoint_manager)
        
        with pytest.raises(ValueError, match="Unknown stage.*invalid_stage"):
            result.get_stage_data("invalid_stage")
    
    def test_lazy_loading_properties(self, job_id, checkpoint_manager, sample_audio_data):
        """Test lazy loading properties."""
        result = PipelineResult(job_id, checkpoint_manager)
        
        # Define mock return values for each stage
        mock_returns = [
            sample_audio_data,  # preprocessor
            {"vocals": np.array([1, 2, 3])},  # separator
            [{"speaker": "A", "start": 0, "end": 10}],  # diarizer
            {"text": "hello world", "segments": []},  # recognizer
            {"audio": np.array([7, 8, 9])},  # synthesizer
            np.array([10, 11, 12])  # reconstructor
        ]
        
        checkpoint_manager.load_stage_data = Mock(side_effect=mock_returns)
        
        # Test original_audio property
        assert result.original_audio is not None
        
        # Test separated_audio property  
        assert result.separated_audio is not None
        assert "vocals" in result.separated_audio
        
        # Test speaker_segments property
        assert result.speaker_segments is not None
        assert len(result.speaker_segments) == 1
        
        # Test transcription property
        assert result.transcription is not None
        assert "text" in result.transcription
        
        # Test synthesized_audio property
        assert result.synthesized_audio is not None
        assert "audio" in result.synthesized_audio
        
        # Test final_audio property
        assert result.final_audio is not None
        
        # Verify checkpoint manager was called appropriately
        assert checkpoint_manager.load_stage_data.call_count == 6
    
    def test_caching_behavior(self, job_id, checkpoint_manager, sample_audio_data):
        """Test that data is cached after first load."""
        result = PipelineResult(job_id, checkpoint_manager)
        
        checkpoint_manager.load_stage_data = Mock(return_value=sample_audio_data)
        
        # Access original_audio twice
        audio1, sr1 = result.original_audio, result.sample_rate
        audio2, sr2 = result.original_audio, result.sample_rate
        
        # Should be same objects (cached)
        assert audio1 is audio2
        assert sr1 is sr2
        
        # Should only call load once
        checkpoint_manager.load_stage_data.assert_called_once()
    
    def test_save_stage_data_updates_cache(self, job_id, checkpoint_manager, sample_audio_data):
        """Test that saving stage data updates internal cache."""
        result = PipelineResult(job_id, checkpoint_manager)
        
        checkpoint_manager.save_stage_data = Mock(return_value=Mock())
        
        # Save preprocessor data
        result.save_stage_data("preprocessor", sample_audio_data)
        
        # Should be cached internally
        assert result._original_audio is not None
        assert result._sample_rate is not None
        
        # Accessing properties should not trigger load
        checkpoint_manager.load_stage_data = Mock()
        _ = result.original_audio
        checkpoint_manager.load_stage_data.assert_not_called()
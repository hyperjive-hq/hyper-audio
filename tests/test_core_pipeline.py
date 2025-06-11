"""Tests for core pipeline orchestration."""

import pytest
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.hyper_audio.pipeline.core import ResilientAudioPipeline
from src.hyper_audio.pipeline.models import PipelineState, StageMetrics, PipelineResult
from src.hyper_audio.pipeline.constants import StageStatus, JobStatus, PIPELINE_STAGES
from src.hyper_audio.pipeline.checkpoint import CheckpointManager, StateManager


class TestResilientAudioPipeline:
    """Test the main pipeline orchestrator."""
    
    @pytest.fixture
    def pipeline(self, temp_dir):
        """Create a pipeline instance for testing."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            with patch('src.hyper_audio.pipeline.core.ResilientAudioPipeline._initialize_stages'):
                pipeline = ResilientAudioPipeline(checkpoint_dir=temp_dir / "checkpoints")
                pipeline.stages = {}  # Initialize empty for mocking
                return pipeline
    
    def test_init(self, temp_dir):
        """Test pipeline initialization."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cuda"
            
            with patch('src.hyper_audio.pipeline.core.ResilientAudioPipeline._initialize_stages') as mock_init:
                pipeline = ResilientAudioPipeline(
                    checkpoint_dir=temp_dir / "test_checkpoints",
                    max_retries=5,
                    config={"test": "value"}
                )
                
                assert pipeline.checkpoint_dir == temp_dir / "test_checkpoints"
                assert pipeline.max_retries == 5
                assert pipeline.config["test"] == "value"
                assert pipeline.device == "cuda"
                mock_init.assert_called_once()
    
    def test_init_default_values(self, temp_dir):
        """Test pipeline initialization with default values."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            with patch('src.hyper_audio.pipeline.core.ResilientAudioPipeline._initialize_stages'):
                pipeline = ResilientAudioPipeline()
                
                assert pipeline.max_retries == 3  # DEFAULT_MAX_RETRIES
                assert pipeline.config == {}
                assert "checkpoints" in str(pipeline.checkpoint_dir)
    
    def test_generate_job_id(self, pipeline, sample_audio_file):
        """Test job ID generation."""
        job_id = pipeline._generate_job_id(sample_audio_file)
        
        assert sample_audio_file.stem in job_id
        assert "_" in job_id  # timestamp separator
        assert len(job_id.split("_")) >= 3  # name_date_time
    
    @pytest.mark.asyncio
    async def test_process_audio_success(self, pipeline, sample_audio_file, temp_dir, mock_pipeline_stages, mock_audio_utils):
        """Test successful audio processing."""
        pipeline.stages = mock_pipeline_stages
        output_path = temp_dir / "output.wav"
        
        # Mock all the helper functions
        with patch('src.hyper_audio.pipeline.core.validate_input_path', return_value=sample_audio_file), \
             patch('src.hyper_audio.pipeline.core.validate_output_path', return_value=output_path), \
             patch('src.hyper_audio.pipeline.core.load_or_create_state') as mock_load_state, \
             patch('src.hyper_audio.pipeline.core.save_final_result') as mock_save_final:
            
            # Mock state
            mock_state = Mock(spec=PipelineState)
            mock_state.current_stage = 0
            mock_state.mark_stage_completed = Mock()
            mock_state.to_dict = Mock(return_value={})
            mock_load_state.return_value = mock_state
            
            # Mock stage execution to always succeed
            with patch.object(pipeline, '_execute_stage_with_retry', return_value=True) as mock_execute:
                
                result = await pipeline.process_audio(
                    input_path=sample_audio_file,
                    output_path=output_path,
                    job_id="test_job"
                )
                
                assert isinstance(result, PipelineResult)
                assert result.job_id == "test_job"
                
                # Verify all stages were executed
                assert mock_execute.call_count == len(PIPELINE_STAGES)
                
                # Verify state was updated for each stage
                assert mock_state.mark_stage_completed.call_count == len(PIPELINE_STAGES)
                
                # Verify final result was saved
                mock_save_final.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_process_audio_with_resume(self, pipeline, sample_audio_file, temp_dir, mock_pipeline_stages):
        """Test audio processing with resume from checkpoint."""
        pipeline.stages = mock_pipeline_stages
        output_path = temp_dir / "output.wav"
        
        with patch('src.hyper_audio.pipeline.core.validate_input_path', return_value=sample_audio_file), \
             patch('src.hyper_audio.pipeline.core.validate_output_path', return_value=output_path), \
             patch('src.hyper_audio.pipeline.core.load_or_create_state') as mock_load_state, \
             patch('src.hyper_audio.pipeline.core.save_final_result'):
            
            # Mock state showing some stages already completed
            mock_state = Mock(spec=PipelineState)
            mock_state.current_stage = 3  # Already completed 3 stages
            mock_state.mark_stage_completed = Mock()
            mock_state.to_dict = Mock(return_value={})
            mock_load_state.return_value = mock_state
            
            with patch.object(pipeline, '_execute_stage_with_retry', return_value=True) as mock_execute:
                
                await pipeline.process_audio(
                    input_path=sample_audio_file,
                    output_path=output_path,
                    job_id="test_job",
                    resume_from_checkpoint=True
                )
                
                # Should only execute remaining stages (6 total - 3 completed = 3 remaining)
                assert mock_execute.call_count == len(PIPELINE_STAGES) - 3
    
    @pytest.mark.asyncio
    async def test_process_audio_stage_failure(self, pipeline, sample_audio_file, temp_dir, mock_pipeline_stages):
        """Test audio processing with stage failure."""
        pipeline.stages = mock_pipeline_stages
        output_path = temp_dir / "output.wav"
        
        with patch('src.hyper_audio.pipeline.core.validate_input_path', return_value=sample_audio_file), \
             patch('src.hyper_audio.pipeline.core.validate_output_path', return_value=output_path), \
             patch('src.hyper_audio.pipeline.core.load_or_create_state') as mock_load_state, \
             patch('src.hyper_audio.pipeline.core.save_failure_report') as mock_save_failure:
            
            mock_state = Mock(spec=PipelineState)
            mock_state.current_stage = 0
            mock_state.to_dict = Mock(return_value={})
            mock_load_state.return_value = mock_state
            
            # Mock first stage to fail
            with patch.object(pipeline, '_execute_stage_with_retry', return_value=False):
                
                with pytest.raises(RuntimeError, match="Stage .* failed after .* retries"):
                    await pipeline.process_audio(
                        input_path=sample_audio_file,
                        output_path=output_path,
                        job_id="test_job"
                    )
                
                # Should save failure report (gets called twice - once for stage failure, once for exception)
                assert mock_save_failure.call_count == 2
    
    @pytest.mark.asyncio
    async def test_execute_stage_with_retry_success(self, pipeline, mock_pipeline_stages, mock_torch):
        """Test successful stage execution with retry logic."""
        pipeline.stages = mock_pipeline_stages
        
        # Mock dependencies
        mock_state = Mock(spec=PipelineState)
        mock_state.add_stage_metrics = Mock()
        mock_result = Mock(spec=PipelineResult)
        mock_result.save_stage_data = Mock(return_value=Path("/test/checkpoint.pkl"))
        mock_state_manager = Mock(spec=StateManager)
        mock_state_manager.save_state = Mock()
        
        # Mock stage execution to return test data
        mock_pipeline_stages["preprocessor"].process.return_value = ("test_data", 16000)
        
        with patch.object(pipeline, '_execute_single_stage', return_value="test_output") as mock_execute:
            
            success = await pipeline._execute_stage_with_retry(
                stage_name="preprocessor",
                stage_idx=0,
                state=mock_state,
                result=mock_result,
                state_manager=mock_state_manager,
                target_speaker=None,
                replacement_voice=None,
                progress_callback=None
            )
            
            assert success is True
            mock_execute.assert_called_once()
            mock_result.save_stage_data.assert_called_once()
            mock_state.add_stage_metrics.assert_called_once()
            mock_state_manager.save_state.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_execute_stage_with_retry_failure(self, pipeline, mock_pipeline_stages, mock_torch):
        """Test stage execution with retry after failures."""
        pipeline.stages = mock_pipeline_stages
        pipeline.max_retries = 2
        
        mock_state = Mock(spec=PipelineState)
        mock_state.add_stage_metrics = Mock()
        mock_result = Mock(spec=PipelineResult)
        mock_state_manager = Mock(spec=StateManager)
        mock_state_manager.save_state = Mock()
        
        # Mock stage execution to always fail
        with patch.object(pipeline, '_execute_single_stage', side_effect=RuntimeError("Test error")):
            
            success = await pipeline._execute_stage_with_retry(
                stage_name="preprocessor",
                stage_idx=0,
                state=mock_state,
                result=mock_result,
                state_manager=mock_state_manager,
                target_speaker=None,
                replacement_voice=None,
                progress_callback=None
            )
            
            assert success is False
            # Should add failure metrics for each attempt
            assert mock_state.add_stage_metrics.call_count == pipeline.max_retries + 1
    
    @pytest.mark.asyncio
    async def test_execute_stage_with_retry_eventual_success(self, pipeline, mock_pipeline_stages, mock_torch):
        """Test stage execution that fails then succeeds."""
        pipeline.stages = mock_pipeline_stages
        pipeline.max_retries = 2
        
        mock_state = Mock(spec=PipelineState)
        mock_state.add_stage_metrics = Mock()
        mock_result = Mock(spec=PipelineResult)
        mock_result.save_stage_data = Mock(return_value=Path("/test/checkpoint.pkl"))
        mock_state_manager = Mock(spec=StateManager)
        mock_state_manager.save_state = Mock()
        
        # Mock stage execution to fail twice then succeed
        call_count = 0
        def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count <= 2:
                raise RuntimeError("Test error")
            return "test_output"
        
        with patch.object(pipeline, '_execute_single_stage', side_effect=side_effect):
            
            success = await pipeline._execute_stage_with_retry(
                stage_name="preprocessor",
                stage_idx=0,
                state=mock_state,
                result=mock_result,
                state_manager=mock_state_manager,
                target_speaker=None,
                replacement_voice=None,
                progress_callback=None
            )
            
            assert success is True
            # Should add failure metrics for failed attempts + success metrics
            assert mock_state.add_stage_metrics.call_count == 3  # 2 failures + 1 success
    
    @pytest.mark.asyncio 
    async def test_execute_single_stage_preprocessor(self, pipeline, mock_pipeline_stages):
        """Test executing preprocessor stage."""
        pipeline.stages = mock_pipeline_stages
        
        mock_result = Mock(spec=PipelineResult)
        mock_result.job_id = "test_job"
        
        expected_output = ("audio_data", 16000)
        mock_pipeline_stages["preprocessor"].process.return_value = expected_output
        
        output = await pipeline._execute_single_stage(
            stage_name="preprocessor",
            stage=mock_pipeline_stages["preprocessor"],
            result=mock_result,
            target_speaker=None,
            replacement_voice=None
        )
        
        assert output == expected_output
        mock_pipeline_stages["preprocessor"].process.assert_called_once_with("test_job")
    
    @pytest.mark.asyncio
    async def test_execute_single_stage_separator(self, pipeline, mock_pipeline_stages):
        """Test executing separator stage."""
        pipeline.stages = mock_pipeline_stages
        
        mock_result = Mock(spec=PipelineResult)
        mock_result.original_audio = "audio_data"
        mock_result.sample_rate = 16000
        
        expected_output = {"vocals": "vocal_audio", "music": "music_audio"}
        mock_pipeline_stages["separator"].process.return_value = expected_output
        
        output = await pipeline._execute_single_stage(
            stage_name="separator",
            stage=mock_pipeline_stages["separator"],
            result=mock_result,
            target_speaker=None,
            replacement_voice=None
        )
        
        assert output == expected_output
        mock_pipeline_stages["separator"].process.assert_called_once_with("audio_data", 16000)
    
    def test_get_job_status(self, pipeline):
        """Test getting job status."""
        job_id = "test_job"
        expected_status = {"job_id": job_id, "status": "completed"}
        
        with patch('src.hyper_audio.pipeline.analytics_simple.PipelineAnalytics') as mock_analytics_class:
            mock_analytics = Mock()
            mock_analytics.get_job_status.return_value = expected_status
            mock_analytics_class.return_value = mock_analytics
            
            status = pipeline.get_job_status(job_id)
            
            assert status == expected_status
            mock_analytics.get_job_status.assert_called_once_with(job_id)
    
    def test_list_jobs(self, pipeline):
        """Test listing jobs."""
        expected_jobs = [
            {"job_id": "job1", "status": "completed"},
            {"job_id": "job2", "status": "in_progress"}
        ]
        
        with patch('src.hyper_audio.pipeline.analytics_simple.PipelineAnalytics') as mock_analytics_class:
            mock_analytics = Mock()
            mock_analytics.list_jobs.return_value = expected_jobs
            mock_analytics_class.return_value = mock_analytics
            
            jobs = pipeline.list_jobs(limit=10)
            
            assert jobs == expected_jobs
            mock_analytics.list_jobs.assert_called_once_with(10)
    
    def test_get_stage_info(self, pipeline, mock_pipeline_stages):
        """Test getting stage information."""
        pipeline.stages = mock_pipeline_stages
        
        with patch('src.hyper_audio.pipeline.core.get_stage_info') as mock_get_info:
            expected_info = {"stage1": {"status": "initialized"}}
            mock_get_info.return_value = expected_info
            
            info = pipeline.get_stage_info()
            
            assert info == expected_info
            mock_get_info.assert_called_once_with(mock_pipeline_stages)
    
    @pytest.mark.asyncio
    async def test_cleanup_job(self, pipeline, temp_dir):
        """Test cleaning up job."""
        job_id = "test_job"
        job_dir = pipeline.checkpoint_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)
        
        # Create some test files
        (job_dir / "test_file.txt").write_text("test")
        
        assert job_dir.exists()
        
        await pipeline.cleanup_job(job_id)
        
        assert not job_dir.exists()
    
    @pytest.mark.asyncio
    async def test_cleanup_nonexistent_job(self, pipeline):
        """Test cleaning up non-existent job."""
        job_id = "nonexistent_job"
        
        # Should not raise an error
        await pipeline.cleanup_job(job_id)
    
    @pytest.mark.asyncio
    async def test_cleanup_pipeline(self, pipeline, mock_pipeline_stages):
        """Test cleaning up pipeline resources."""
        pipeline.stages = mock_pipeline_stages
        
        with patch('src.hyper_audio.pipeline.core.cleanup_pipeline_resources') as mock_cleanup:
            
            await pipeline.cleanup()
            
            mock_cleanup.assert_called_once_with(mock_pipeline_stages)
    
    @pytest.mark.asyncio
    async def test_progress_callback(self, pipeline, sample_audio_file, temp_dir, mock_pipeline_stages):
        """Test progress callback functionality."""
        pipeline.stages = mock_pipeline_stages
        output_path = temp_dir / "output.wav"
        
        progress_calls = []
        def progress_callback(message, current, total):
            progress_calls.append((message, current, total))
        
        with patch('src.hyper_audio.pipeline.core.validate_input_path', return_value=sample_audio_file), \
             patch('src.hyper_audio.pipeline.core.validate_output_path', return_value=output_path), \
             patch('src.hyper_audio.pipeline.core.load_or_create_state') as mock_load_state, \
             patch('src.hyper_audio.pipeline.core.save_final_result'):
            
            mock_state = Mock(spec=PipelineState)
            mock_state.current_stage = 0
            mock_state.mark_stage_completed = Mock()
            mock_state.to_dict = Mock(return_value={})
            mock_load_state.return_value = mock_state
            
            async def mock_execute_stage_with_retry(stage_name, stage_idx, state, result, state_manager, target_speaker, replacement_voice, progress_callback):
                if progress_callback:
                    progress_callback(f"Running {stage_name} (attempt 1)", stage_idx, len(PIPELINE_STAGES))
                return True
            
            with patch.object(pipeline, '_execute_stage_with_retry', side_effect=mock_execute_stage_with_retry):
                
                await pipeline.process_audio(
                    input_path=sample_audio_file,
                    output_path=output_path,
                    progress_callback=progress_callback
                )
                
                # Should have progress calls for each stage + completion
                assert len(progress_calls) >= len(PIPELINE_STAGES)
                
                # Check final completion call
                final_call = progress_calls[-1]
                assert "complete" in final_call[0].lower()
                assert final_call[1] == final_call[2]  # current == total
    
    @pytest.mark.asyncio
    async def test_input_validation(self, pipeline, temp_dir):
        """Test input validation."""
        nonexistent_file = temp_dir / "nonexistent.wav"
        output_path = temp_dir / "output.wav"
        
        with patch('src.hyper_audio.pipeline.core.validate_input_path', side_effect=FileNotFoundError("File not found")):
            
            with pytest.raises(FileNotFoundError):
                await pipeline.process_audio(
                    input_path=nonexistent_file,
                    output_path=output_path
                )
    

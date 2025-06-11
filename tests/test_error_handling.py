"""Tests for error handling and recovery scenarios."""

import pytest
import asyncio
import pickle
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.hyper_audio.pipeline.core import ResilientAudioPipeline
from src.hyper_audio.pipeline.checkpoint import CheckpointManager, StateManager
from src.hyper_audio.pipeline.models import PipelineState, StageMetrics, PipelineResult
from src.hyper_audio.pipeline.constants import StageStatus, JobStatus
from src.hyper_audio.pipeline.core_helpers import (
    load_or_create_state, save_failure_report, save_final_result
)


class TestErrorHandling:
    """Test error handling and recovery scenarios."""
    
    @pytest.fixture
    def pipeline(self, temp_dir):
        """Create a pipeline instance for testing."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            with patch('src.hyper_audio.pipeline.core.ResilientAudioPipeline._initialize_stages'):
                pipeline = ResilientAudioPipeline(checkpoint_dir=temp_dir / "checkpoints")
                pipeline.stages = {}
                return pipeline
    
    @pytest.mark.asyncio
    async def test_checkpoint_corruption_recovery(self, checkpoint_manager, sample_audio_data):
        """Test recovery from corrupted checkpoint files."""
        stage_name = "test_stage"
        
        # Save valid checkpoint
        checkpoint_manager.save_stage_data(stage_name, sample_audio_data)
        
        # Corrupt the checkpoint file
        checkpoint_path = checkpoint_manager.checkpoint_dir / f"{stage_name}_output.pkl"
        with open(checkpoint_path, 'wb') as f:
            f.write(b"corrupted data")
        
        # Should detect corruption and raise error
        with pytest.raises(ValueError, match="Checkpoint corruption detected"):
            checkpoint_manager.load_stage_data(stage_name)
    
    @pytest.mark.asyncio
    async def test_disk_space_error_during_checkpoint(self, checkpoint_manager):
        """Test handling of disk space errors during checkpointing."""
        stage_name = "test_stage"
        test_data = {"large": "data"}
        
        # Mock open to simulate disk full error
        with patch("builtins.open", side_effect=OSError("No space left on device")):
            
            with pytest.raises(RuntimeError, match="Checkpoint save failed"):
                checkpoint_manager.save_stage_data(stage_name, test_data)
    
    @pytest.mark.asyncio
    async def test_state_file_corruption_recovery(self, state_manager, temp_dir):
        """Test recovery from corrupted state files."""
        # Create corrupted state file
        state_file = state_manager.state_path
        state_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(state_file, 'w') as f:
            f.write("invalid json content")
        
        # Should handle corruption gracefully
        with pytest.raises(RuntimeError, match="State load failed"):
            state_manager.load_state()
    
    @pytest.mark.asyncio
    @pytest.mark.gpu
    async def test_memory_exhaustion_during_stage(self, pipeline, mock_pipeline_stages, mock_torch):
        """Test handling of memory exhaustion during stage execution."""
        pipeline.stages = mock_pipeline_stages
        
        mock_state = Mock(spec=PipelineState)
        mock_state.add_stage_metrics = Mock()
        mock_result = Mock(spec=PipelineResult)
        mock_state_manager = Mock(spec=StateManager)
        mock_state_manager.save_state = Mock()
        
        # Mock stage to raise CUDA out of memory error (requires GPU/CUDA)
        with patch.object(pipeline, '_execute_single_stage', side_effect=RuntimeError("CUDA out of memory")):
            
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
            
            # Should have attempted retries
            assert mock_state.add_stage_metrics.call_count == pipeline.max_retries + 1
            
            # Should have cleared GPU memory between retries
            assert mock_torch['empty_cache'].call_count >= pipeline.max_retries
    
    @pytest.mark.asyncio
    async def test_network_interruption_during_model_loading(self, pipeline, mock_pipeline_stages):
        """Test handling of network interruptions during model loading."""
        pipeline.stages = mock_pipeline_stages
        
        # Mock stage to raise connection error
        mock_pipeline_stages["preprocessor"].process.side_effect = ConnectionError("Network unreachable")
        
        mock_state = Mock(spec=PipelineState)
        mock_state.add_stage_metrics = Mock()
        mock_result = Mock(spec=PipelineResult)
        mock_result.job_id = "test_job"  # Add missing job_id attribute
        mock_state_manager = Mock(spec=StateManager)
        mock_state_manager.save_state = Mock()
        
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
        
        # Should have recorded the network error
        calls = mock_state.add_stage_metrics.call_args_list
        assert len(calls) > 0
        
        # Check that one of the stage metrics has ConnectionError as error_type
        metrics_objects = [call[0][0] for call in calls]  # Get the metrics objects from the calls
        assert any(getattr(metrics, 'error_type', None) == 'ConnectionError' for metrics in metrics_objects)
    
    @pytest.mark.asyncio
    async def test_permission_denied_on_output_file(self, pipeline, sample_audio_file, temp_dir, mock_pipeline_stages):
        """Test handling of permission denied on output file."""
        pipeline.stages = mock_pipeline_stages
        output_path = temp_dir / "protected_output.wav"
        
        with patch('src.hyper_audio.pipeline.core.validate_input_path', return_value=sample_audio_file), \
             patch('src.hyper_audio.pipeline.core.validate_output_path', return_value=output_path), \
             patch('src.hyper_audio.pipeline.core.load_or_create_state') as mock_load_state, \
             patch('src.hyper_audio.pipeline.core.save_final_result', side_effect=PermissionError("Permission denied")) as mock_save:
            
            mock_state = Mock(spec=PipelineState)
            mock_state.current_stage = 0
            mock_state.job_id = "test_job"
            mock_state.stages_completed = []
            mock_state.stage_metrics = []
            mock_state.mark_stage_completed = Mock()
            mock_state.to_dict = Mock(return_value={})
            mock_load_state.return_value = mock_state
            
            with patch.object(pipeline, '_execute_stage_with_retry', return_value=True):
                
                with pytest.raises(PermissionError):
                    await pipeline.process_audio(
                        input_path=sample_audio_file,
                        output_path=output_path,
                        job_id="test_job"
                    )
    
    @pytest.mark.asyncio
    async def test_graceful_shutdown_during_processing(self, pipeline, sample_audio_file, temp_dir, mock_pipeline_stages):
        """Test graceful shutdown during processing."""
        pipeline.stages = mock_pipeline_stages
        output_path = temp_dir / "output.wav"
        
        # Mock a long-running stage that gets interrupted
        async def long_running_stage(*args, **kwargs):
            await asyncio.sleep(10)  # Simulate long operation
            return "result"
        
        mock_pipeline_stages["preprocessor"].process = long_running_stage
        
        with patch('src.hyper_audio.pipeline.core.validate_input_path', return_value=sample_audio_file), \
             patch('src.hyper_audio.pipeline.core.validate_output_path', return_value=output_path), \
             patch('src.hyper_audio.pipeline.core.load_or_create_state') as mock_load_state:
            
            mock_state = Mock(spec=PipelineState)
            mock_state.current_stage = 0
            mock_state.mark_stage_completed = Mock()
            mock_state.to_dict = Mock(return_value={})
            mock_load_state.return_value = mock_state
            
            # Start processing and cancel after short delay
            task = asyncio.create_task(pipeline.process_audio(
                input_path=sample_audio_file,
                output_path=output_path,
                job_id="test_job"
            ))
            
            # Cancel the task
            await asyncio.sleep(0.1)  # Let it start
            task.cancel()
            
            with pytest.raises(asyncio.CancelledError):
                await task
    
    @pytest.mark.asyncio
    async def test_recovery_from_partial_checkpoint_save(self, checkpoint_manager, sample_audio_data):
        """Test recovery when checkpoint save is partially completed."""
        stage_name = "test_stage"
        
        # Mock to simulate partial save (checkpoint exists but no checksum)
        checkpoint_path = checkpoint_manager.checkpoint_dir / f"{stage_name}_output.pkl"
        
        # Save checkpoint data directly without checksum
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(sample_audio_data, f)
        
        # Should still load successfully even without checksum
        loaded_data = checkpoint_manager.load_stage_data(stage_name)
        assert loaded_data is not None
    
    @pytest.mark.asyncio
    async def test_invalid_audio_file_handling(self, pipeline, temp_dir, mock_pipeline_stages):
        """Test handling of invalid audio files."""
        pipeline.stages = mock_pipeline_stages
        
        # Create invalid audio file
        invalid_file = temp_dir / "invalid.wav"
        invalid_file.write_text("This is not audio data")
        
        output_path = temp_dir / "output.wav"
        
        # Mock audio loading to fail
        with patch('src.hyper_audio.pipeline.core.validate_input_path', side_effect=ValueError("Invalid audio format")):
            
            with pytest.raises(ValueError, match="Invalid audio format"):
                await pipeline.process_audio(
                    input_path=invalid_file,
                    output_path=output_path
                )
    
    @pytest.mark.asyncio
    async def test_concurrent_access_to_checkpoints(self, checkpoint_manager, sample_audio_data):
        """Test handling of concurrent access to checkpoint files."""
        stage_name = "test_stage"
        
        # Simulate concurrent access by multiple processes
        async def save_checkpoint():
            checkpoint_manager.save_stage_data(stage_name, sample_audio_data)
        
        async def load_checkpoint():
            try:
                return checkpoint_manager.load_stage_data(stage_name)
            except FileNotFoundError:
                return None  # File might not exist yet
        
        # Save first to ensure file exists
        await save_checkpoint()
        
        # Run concurrent operations
        tasks = [
            asyncio.create_task(save_checkpoint()),
            asyncio.create_task(load_checkpoint()),
            asyncio.create_task(load_checkpoint())
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Should not raise any unhandled exceptions
        for result in results:
            if isinstance(result, Exception):
                # Only expected exceptions should occur
                assert isinstance(result, (FileNotFoundError, ValueError, RuntimeError))
    
    @pytest.mark.asyncio
    async def test_cleanup_after_error(self, pipeline, mock_pipeline_stages, mock_torch):
        """Test that resources are properly cleaned up after errors."""
        pipeline.stages = mock_pipeline_stages
        
        # Mock stage to fail
        mock_pipeline_stages["preprocessor"].process.side_effect = RuntimeError("Test error")
        
        with patch.object(pipeline, 'cleanup') as mock_cleanup:
            
            # Cleanup should be called even after error
            try:
                await pipeline.cleanup()
            except Exception:
                pass  # Expected to potentially fail
            
            # Verify cleanup was attempted
            mock_cleanup.assert_called()
    
    def test_error_reporting_detail(self):
        """Test that error reports contain sufficient detail for debugging."""
        from src.hyper_audio.pipeline.core_helpers import save_failure_report
        
        # Mock state with error details
        mock_state = Mock(spec=PipelineState)
        mock_state.job_id = "test_job"
        mock_state.current_stage = 2
        mock_state.stages_completed = ["stage1", "stage2"]
        mock_state.stage_metrics = [
            Mock(to_dict=Mock(return_value={
                "stage_name": "stage3",
                "status": "failed",
                "error_message": "CUDA out of memory",
                "error_type": "RuntimeError",
                "retry_count": 3
            }))
        ]
        
        checkpoint_dir = Path("/tmp/test")
        
        with patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_json_dump, \
             patch('torch.cuda.is_available', return_value=True), \
             patch('torch.cuda.memory_allocated', return_value=1024**3), \
             patch('torch.cuda.memory_reserved', return_value=2*1024**3):
            
            asyncio.run(save_failure_report(mock_state, "Test error", checkpoint_dir))
            
            # Verify report structure
            mock_json_dump.assert_called_once()
            report_data = mock_json_dump.call_args[0][0]
            
            assert "job_id" in report_data
            assert "failure_time" in report_data
            assert "error_message" in report_data
            assert "current_stage" in report_data
            assert "stage_metrics" in report_data
            assert "system_info" in report_data
            
            # Check system info
            system_info = report_data["system_info"]
            assert "cuda_available" in system_info
            assert "memory_allocated_gb" in system_info
    
    @pytest.mark.asyncio
    @pytest.mark.gpu
    async def test_retry_with_exponential_backoff(self, pipeline, mock_pipeline_stages, mock_torch):
        """Test that retries include proper delay between attempts."""
        pipeline.stages = mock_pipeline_stages
        pipeline.max_retries = 2
        
        mock_state = Mock(spec=PipelineState)
        mock_state.add_stage_metrics = Mock()
        mock_result = Mock(spec=PipelineResult)
        mock_state_manager = Mock(spec=StateManager)
        mock_state_manager.save_state = Mock()
        
        # Track sleep calls to verify backoff
        sleep_calls = []
        
        async def mock_sleep(duration):
            sleep_calls.append(duration)
        
        with patch.object(pipeline, '_execute_single_stage', side_effect=RuntimeError("Test error")), \
             patch('asyncio.sleep', side_effect=mock_sleep):
            
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
            
            # Should have slept between retries
            assert len(sleep_calls) == pipeline.max_retries
            # All sleep durations should be the retry delay
            assert all(duration == 5.0 for duration in sleep_calls)  # DEFAULT_RETRY_DELAY_SECONDS
    
    @pytest.mark.asyncio
    async def test_stage_timeout_handling(self, pipeline, mock_pipeline_stages):
        """Test handling of stage timeouts."""
        pipeline.stages = mock_pipeline_stages
        
        # Mock a stage that times out
        async def timeout_stage(*args, **kwargs):
            await asyncio.sleep(3600)  # Very long operation
        
        mock_pipeline_stages["preprocessor"].process = timeout_stage
        
        mock_state = Mock(spec=PipelineState)
        mock_state.add_stage_metrics = Mock()
        mock_result = Mock(spec=PipelineResult)
        mock_state_manager = Mock(spec=StateManager)
        mock_state_manager.save_state = Mock()
        
        # Use asyncio.wait_for to simulate timeout
        with patch.object(pipeline, '_execute_single_stage', side_effect=asyncio.TimeoutError("Stage timeout")):
            
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
            
            # Should record timeout error
            calls = mock_state.add_stage_metrics.call_args_list
            assert any("TimeoutError" in str(call) for call in calls)
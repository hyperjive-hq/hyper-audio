"""Integration tests for the complete pipeline."""

import pytest
import asyncio
import numpy as np
from pathlib import Path
from unittest.mock import Mock, AsyncMock, patch

from src.hyper_audio.pipeline import ResilientAudioPipeline, PipelineAnalytics
from src.hyper_audio.pipeline.constants import StageStatus, JobStatus
from src.hyper_audio.pipeline.models import PipelineState, StageMetrics


class TestPipelineIntegration:
    """Integration tests for the complete pipeline system."""
    
    @pytest.fixture
    def mock_stages(self):
        """Create mock stages that simulate real pipeline behavior."""
        stages = {}
        
        # Preprocessor: loads audio and returns processed data
        preprocessor = AsyncMock()
        preprocessor.process = AsyncMock(return_value=(
            np.random.rand(16000).astype(np.float32),  # 1 second of audio
            16000  # sample rate
        ))
        stages["preprocessor"] = preprocessor
        
        # Separator: separates vocals from music
        separator = AsyncMock()
        separator.process = AsyncMock(return_value={
            "vocals": np.random.rand(16000).astype(np.float32),
            "music": np.random.rand(16000).astype(np.float32)
        })
        stages["separator"] = separator
        
        # Diarizer: identifies speakers
        diarizer = AsyncMock()
        diarizer.process = AsyncMock(return_value=[
            {"speaker": "Speaker_A", "start": 0.0, "end": 5.0},
            {"speaker": "Speaker_B", "start": 5.0, "end": 10.0}
        ])
        stages["diarizer"] = diarizer
        
        # Recognizer: transcribes speech
        recognizer = AsyncMock()
        recognizer.process = AsyncMock(return_value={
            "text": "Hello world, this is a test transcription.",
            "segments": [
                {"text": "Hello world", "start": 0.0, "end": 2.5, "speaker": "Speaker_A"},
                {"text": "this is a test transcription", "start": 5.0, "end": 8.5, "speaker": "Speaker_B"}
            ]
        })
        stages["recognizer"] = recognizer
        
        # Synthesizer: generates new voice
        synthesizer = AsyncMock()
        synthesizer.process = AsyncMock(return_value={
            "Speaker_A": np.random.rand(8000).astype(np.float32),  # Half the audio
        })
        stages["synthesizer"] = synthesizer
        
        # Reconstructor: combines everything
        reconstructor = AsyncMock()
        reconstructor.process = AsyncMock(return_value=
            np.random.rand(16000).astype(np.float32)  # Final audio
        )
        stages["reconstructor"] = reconstructor
        
        return stages
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_complete_pipeline_flow(self, temp_dir, sample_audio_file, mock_stages, mock_audio_utils):
        """Test complete pipeline from start to finish."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            # Create pipeline
            pipeline = ResilientAudioPipeline(checkpoint_dir=temp_dir / "checkpoints")
            pipeline.stages = mock_stages
            
            output_path = temp_dir / "output.wav"
            
            # Run complete pipeline
            result = await pipeline.process_audio(
                input_path=sample_audio_file,
                output_path=output_path,
                job_id="integration_test",
                target_speaker="Speaker_A"
            )
            
            # Verify all stages were called
            for stage_name, stage in mock_stages.items():
                stage.process.assert_called_once()
            
            # Verify result structure
            assert result.job_id == "integration_test"
            assert result.original_audio is not None
            assert result.separated_audio is not None
            assert result.speaker_segments is not None
            assert result.transcription is not None
            assert result.synthesized_audio is not None
            assert result.final_audio is not None
            
            # Verify final audio was saved
            mock_audio_utils['save_audio'].assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_resume_after_interruption(self, temp_dir, sample_audio_file, mock_stages, mock_audio_utils):
        """Test pipeline resume functionality after interruption."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            # Create pipeline
            pipeline = ResilientAudioPipeline(checkpoint_dir=temp_dir / "checkpoints")
            pipeline.stages = mock_stages
            
            output_path = temp_dir / "output.wav"
            job_id = "resume_test"
            
            # First run: fail at separator stage
            mock_stages["separator"].process.side_effect = RuntimeError("Simulated failure")
            
            with pytest.raises(RuntimeError):
                await pipeline.process_audio(
                    input_path=sample_audio_file,
                    output_path=output_path,
                    job_id=job_id
                )
            
            # Verify preprocessor completed and saved checkpoint
            mock_stages["preprocessor"].process.assert_called_once()
            
            # Second run: fix the failure and resume
            mock_stages["separator"].process.side_effect = None
            mock_stages["separator"].process.reset_mock()
            
            result = await pipeline.process_audio(
                input_path=sample_audio_file,
                output_path=output_path,
                job_id=job_id,
                resume_from_checkpoint=True
            )
            
            # Verify preprocessor was NOT called again (resumed from checkpoint)
            assert mock_stages["preprocessor"].process.call_count == 1  # Still only called once
            
            # Verify separator and subsequent stages were called
            mock_stages["separator"].process.assert_called_once()
            mock_stages["diarizer"].process.assert_called_once()
            
            # Verify successful completion
            assert result.job_id == job_id
            assert result.final_audio is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_pipeline_with_retry_recovery(self, temp_dir, sample_audio_file, mock_stages, mock_audio_utils):
        """Test pipeline retry mechanism recovers from transient failures."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            # Create pipeline with limited retries
            pipeline = ResilientAudioPipeline(
                checkpoint_dir=temp_dir / "checkpoints",
                max_retries=2
            )
            pipeline.stages = mock_stages
            
            output_path = temp_dir / "output.wav"
            
            # Make separator fail twice then succeed
            call_count = 0
            def failing_separator(*args, **kwargs):
                nonlocal call_count
                call_count += 1
                if call_count <= 2:
                    raise RuntimeError("Transient failure")
                return {
                    "vocals": np.random.rand(16000).astype(np.float32),
                    "music": np.random.rand(16000).astype(np.float32)
                }
            
            mock_stages["separator"].process.side_effect = failing_separator
            
            # Should succeed after retries
            result = await pipeline.process_audio(
                input_path=sample_audio_file,
                output_path=output_path,
                job_id="retry_test"
            )
            
            # Verify separator was called 3 times (initial + 2 retries)
            assert mock_stages["separator"].process.call_count == 3
            
            # Verify successful completion
            assert result.final_audio is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_analytics_integration(self, temp_dir, sample_audio_file, mock_stages, mock_audio_utils):
        """Test analytics integration with pipeline execution."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            # Create pipeline
            pipeline = ResilientAudioPipeline(checkpoint_dir=temp_dir / "checkpoints")
            pipeline.stages = mock_stages
            
            # Create analytics
            analytics = PipelineAnalytics(temp_dir / "checkpoints")
            
            output_path = temp_dir / "output.wav"
            job_id = "analytics_test"
            
            # Run pipeline
            result = await pipeline.process_audio(
                input_path=sample_audio_file,
                output_path=output_path,
                job_id=job_id
            )
            
            # Test analytics can read job status
            status = analytics.get_job_status(job_id)
            assert status["job_id"] == job_id
            assert status["status"] == JobStatus.COMPLETED.value
            assert status["progress_percentage"] == 100.0
            
            # Test job listing
            jobs = analytics.list_jobs()
            assert len(jobs) == 1
            assert jobs[0]["job_id"] == job_id
            
            # Test summary report
            report = analytics.generate_summary_report()
            assert report["summary"]["total_jobs"] == 1
            assert report["summary"]["completed_jobs"] == 1
            assert report["summary"]["success_rate"] == 100.0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multiple_concurrent_jobs(self, temp_dir, sample_audio_file, mock_stages, mock_audio_utils):
        """Test running multiple pipeline jobs concurrently."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            # Create multiple pipeline instances
            pipelines = []
            for i in range(3):
                pipeline = ResilientAudioPipeline(
                    checkpoint_dir=temp_dir / f"checkpoints_{i}"
                )
                # Each pipeline needs its own mock stages to avoid conflicts
                pipeline.stages = {
                    name: AsyncMock(**{
                        'process.return_value': stage.process.return_value
                    })
                    for name, stage in mock_stages.items()
                }
                pipelines.append(pipeline)
            
            # Run jobs concurrently
            tasks = []
            for i, pipeline in enumerate(pipelines):
                task = asyncio.create_task(
                    pipeline.process_audio(
                        input_path=sample_audio_file,
                        output_path=temp_dir / f"output_{i}.wav",
                        job_id=f"concurrent_job_{i}"
                    )
                )
                tasks.append(task)
            
            # Wait for all to complete
            results = await asyncio.gather(*tasks)
            
            # Verify all completed successfully
            assert len(results) == 3
            for i, result in enumerate(results):
                assert result.job_id == f"concurrent_job_{i}"
                assert result.final_audio is not None
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_checkpoint_consistency_across_restarts(self, temp_dir, sample_audio_file, mock_stages, mock_audio_utils):
        """Test checkpoint consistency across pipeline restarts."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            checkpoint_dir = temp_dir / "checkpoints"
            job_id = "consistency_test"
            
            # First pipeline instance
            pipeline1 = ResilientAudioPipeline(checkpoint_dir=checkpoint_dir)
            pipeline1.stages = mock_stages
            
            # Run partial pipeline (fail at diarizer)
            mock_stages["diarizer"].process.side_effect = RuntimeError("Simulated failure")
            
            with pytest.raises(RuntimeError):
                await pipeline1.process_audio(
                    input_path=sample_audio_file,
                    output_path=temp_dir / "output.wav",
                    job_id=job_id
                )
            
            # Create new pipeline instance (simulating restart)
            pipeline2 = ResilientAudioPipeline(checkpoint_dir=checkpoint_dir)
            
            # Create new mock stages for second pipeline
            mock_stages2 = {
                name: AsyncMock(**{
                    'process.return_value': stage.process.return_value
                })
                for name, stage in mock_stages.items()
            }
            pipeline2.stages = mock_stages2
            
            # Fix the diarizer issue
            mock_stages2["diarizer"].process.side_effect = None
            
            # Resume from checkpoint
            result = await pipeline2.process_audio(
                input_path=sample_audio_file,
                output_path=temp_dir / "output.wav",
                job_id=job_id,
                resume_from_checkpoint=True
            )
            
            # Verify preprocessor and separator weren't called again
            mock_stages2["preprocessor"].process.assert_not_called()
            mock_stages2["separator"].process.assert_not_called()
            
            # Verify diarizer and subsequent stages were called
            mock_stages2["diarizer"].process.assert_called_once()
            mock_stages2["recognizer"].process.assert_called_once()
            
            # Verify successful completion
            assert result.job_id == job_id
            assert result.final_audio is not None
    
    @pytest.mark.integration
    def test_pipeline_cleanup_integration(self, temp_dir):
        """Test complete pipeline cleanup."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            # Create pipeline with mock stages
            pipeline = ResilientAudioPipeline(checkpoint_dir=temp_dir / "checkpoints")
            
            mock_stages = {}
            for stage_name in ["preprocessor", "separator", "diarizer", "recognizer", "synthesizer", "reconstructor"]:
                mock_stage = Mock()
                mock_stage.cleanup = AsyncMock()
                mock_stages[stage_name] = mock_stage
            
            pipeline.stages = mock_stages
            
            # Create some job directories and files
            job_dirs = []
            for i in range(3):
                job_id = f"cleanup_test_{i}"
                job_dir = pipeline.checkpoint_dir / job_id
                job_dir.mkdir(parents=True, exist_ok=True)
                
                # Create some files
                (job_dir / "test_file.txt").write_text("test")
                (job_dir / "checkpoint.pkl").write_text("fake checkpoint")
                
                job_dirs.append(job_dir)
            
            # Test individual job cleanup
            asyncio.run(pipeline.cleanup_job("cleanup_test_0"))
            assert not job_dirs[0].exists()
            assert job_dirs[1].exists()  # Others should remain
            assert job_dirs[2].exists()
            
            # Test overall pipeline cleanup
            asyncio.run(pipeline.cleanup())
            
            # Verify all stages were cleaned up
            for stage in mock_stages.values():
                stage.cleanup.assert_called_once()
    
    @pytest.mark.integration
    @pytest.mark.slow
    @pytest.mark.asyncio
    @pytest.mark.gpu
    async def test_large_audio_processing(self, temp_dir, mock_stages, mock_audio_utils):
        """Test pipeline with large audio files (requires sufficient memory)."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            # Create large audio file (simulate 10 minutes)
            large_audio_file = temp_dir / "large_audio.wav"
            
            # Mock audio loading to return large data (10 minutes at 16kHz)
            duration_seconds = 600
            sample_rate = 16000
            large_audio_data = np.random.rand(sample_rate * duration_seconds).astype(np.float32)
            mock_audio_utils['load_audio'].return_value = (large_audio_data, sample_rate)
            
            # Update mock stages to handle large data efficiently
            mock_stages["preprocessor"].process.return_value = (large_audio_data, sample_rate)
            mock_stages["separator"].process.return_value = {
                "vocals": large_audio_data,
                "music": large_audio_data
            }
            mock_stages["diarizer"].process.return_value = [
                {"speaker": "Speaker_A", "start": 0.0, "end": duration_seconds / 2},
                {"speaker": "Speaker_B", "start": duration_seconds / 2, "end": duration_seconds}
            ]
            mock_stages["recognizer"].process.return_value = {
                "text": "Long transcription for large audio file",
                "segments": [{"text": "segment", "start": i * 10, "end": (i + 1) * 10, "speaker": "Speaker_A"} for i in range(60)]
            }
            mock_stages["synthesizer"].process.return_value = {"Speaker_A": large_audio_data}
            mock_stages["reconstructor"].process.return_value = large_audio_data
            
            pipeline = ResilientAudioPipeline(checkpoint_dir=temp_dir / "checkpoints")
            pipeline.stages = mock_stages
            
            # Create the file
            large_audio_file.write_text("fake large audio data")
            
            output_path = temp_dir / "large_output.wav"
            
            # Should handle large files without issues
            result = await pipeline.process_audio(
                input_path=large_audio_file,
                output_path=output_path,
                job_id="large_audio_test"
            )
            
            assert result.final_audio is not None
            
            # Verify checkpoints were created for large data
            job_dir = pipeline.checkpoint_dir / "large_audio_test"
            assert job_dir.exists()
            assert len(list(job_dir.glob("*.pkl"))) > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_progress_tracking_integration(self, temp_dir, sample_audio_file, mock_stages, mock_audio_utils):
        """Test progress tracking throughout pipeline execution."""
        with patch('src.hyper_audio.pipeline.core.settings') as mock_settings:
            mock_settings.device = "cpu"
            
            pipeline = ResilientAudioPipeline(checkpoint_dir=temp_dir / "checkpoints")
            pipeline.stages = mock_stages
            
            progress_updates = []
            
            def progress_callback(message, current, total):
                progress_updates.append({
                    "message": message,
                    "current": current,
                    "total": total,
                    "percentage": (current / total) * 100 if total > 0 else 0
                })
            
            output_path = temp_dir / "output.wav"
            
            result = await pipeline.process_audio(
                input_path=sample_audio_file,
                output_path=output_path,
                job_id="progress_test",
                progress_callback=progress_callback
            )
            
            # Verify progress updates
            assert len(progress_updates) > 0
            
            # Check that progress goes from 0 to 100%
            first_update = progress_updates[0]
            last_update = progress_updates[-1]
            
            assert first_update["percentage"] < last_update["percentage"]
            assert last_update["percentage"] == 100.0
            assert "complete" in last_update["message"].lower()
            
            # Verify stage names appear in progress messages
            stage_names = ["preprocessor", "separator", "diarizer", "recognizer", "synthesizer", "reconstructor"]
            progress_messages = [update["message"].lower() for update in progress_updates]
            
            for stage_name in stage_names:
                assert any(stage_name in message for message in progress_messages)
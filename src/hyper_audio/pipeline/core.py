"""Resilient audio processing pipeline with checkpointing and failure recovery."""

import asyncio
import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, timezone
import torch

from ..config.settings import settings
from ..utils.logging_utils import get_logger
from .constants import (
    StageStatus, JobStatus, PIPELINE_STAGES, 
    DEFAULT_MAX_RETRIES, DEFAULT_RETRY_DELAY_SECONDS, DEFAULT_CHECKPOINT_DIR,
    FAILURE_REPORT_FILENAME
)
from .models import PipelineState, PipelineResult, StageMetrics, JobSummary
from .checkpoint import CheckpointManager, StateManager
from .core_helpers import (
    load_or_create_state, save_failure_report, save_final_result,
    get_memory_usage, get_stage_info, cleanup_pipeline_resources,
    validate_input_path, validate_output_path
)

logger = get_logger("pipeline.core")


class ResilientAudioPipeline:
    """Resilient audio processing pipeline with checkpointing and failure recovery."""
    
    def __init__(self, 
                 checkpoint_dir: Optional[Union[str, Path]] = None,
                 max_retries: int = DEFAULT_MAX_RETRIES,
                 config: Optional[Dict[str, Any]] = None):
        """Initialize the resilient audio processing pipeline.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
            max_retries: Maximum number of retries per stage
            config: Optional configuration overrides
        """
        self.config = config or {}
        self.max_retries = max_retries
        self.device = settings.device
        
        # Setup checkpoint directory
        if checkpoint_dir is None:
            checkpoint_dir = Path.cwd() / DEFAULT_CHECKPOINT_DIR
        self.checkpoint_dir = Path(checkpoint_dir)
        
        self.stages = {}
        self._initialize_stages()
        
        logger.info(f"ResilientAudioPipeline initialized with device: {self.device}")
        logger.info(f"Checkpoint directory: {self.checkpoint_dir}")
    
    def _initialize_stages(self):
        """Initialize all pipeline stages lazily."""
        # Import here to avoid circular imports
        from .stages import (
            AudioPreprocessor, VoiceSeparator, SpeakerDiarizer,
            SpeechRecognizer, VoiceSynthesizer, AudioReconstructor
        )
        
        stage_classes = {
            "AudioPreprocessor": AudioPreprocessor,
            "VoiceSeparator": VoiceSeparator,
            "SpeakerDiarizer": SpeakerDiarizer,
            "SpeechRecognizer": SpeechRecognizer,
            "VoiceSynthesizer": VoiceSynthesizer,
            "AudioReconstructor": AudioReconstructor
        }
        
        try:
            for stage_def in PIPELINE_STAGES:
                stage_class = stage_classes[stage_def["class"]]
                self.stages[stage_def["name"]] = stage_class()
            
            logger.info("All pipeline stages initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize pipeline stages: {e}")
            raise
    
    async def process_audio(
        self,
        input_path: Union[str, Path],
        output_path: Union[str, Path],
        job_id: Optional[str] = None,
        resume_from_checkpoint: bool = True,
        target_speaker: Optional[str] = None,
        replacement_voice: Optional[Union[str, Path]] = None,
        progress_callback: Optional[Callable[[str, int, int], None]] = None
    ) -> PipelineResult:
        """Process audio through the complete pipeline with checkpointing.
        
        Args:
            input_path: Path to input audio file
            output_path: Path for output audio file
            job_id: Unique job identifier (auto-generated if None)
            resume_from_checkpoint: Whether to resume from existing checkpoint
            target_speaker: ID of speaker to replace
            replacement_voice: Path to reference voice for synthesis
            progress_callback: Optional callback for progress updates (message, current, total)
            
        Returns:
            PipelineResult with processing results
        """
        # Generate job ID if not provided
        if job_id is None:
            job_id = self._generate_job_id(input_path)
        
        job_checkpoint_dir = self.checkpoint_dir / job_id
        
        # Initialize managers
        checkpoint_manager = CheckpointManager(job_checkpoint_dir)
        state_manager = StateManager(job_checkpoint_dir)
        
        # Validate paths
        input_path = validate_input_path(input_path)
        output_path = validate_output_path(output_path)
        
        # Initialize or load pipeline state
        state = await load_or_create_state(
            job_id, input_path, output_path, state_manager, resume_from_checkpoint, self.config
        )
        
        result = PipelineResult(job_id, checkpoint_manager)
        
        logger.info(f"Starting pipeline job: {job_id}")
        logger.info(f"Input: {input_path} -> Output: {output_path}")
        
        try:
            # Process each stage
            for stage_idx, stage_def in enumerate(PIPELINE_STAGES):
                stage_name = stage_def["name"]
                
                if stage_idx < state.current_stage:
                    logger.info(f"Skipping completed stage: {stage_name}")
                    continue
                
                # Execute stage with retry logic
                success = await self._execute_stage_with_retry(
                    stage_name, stage_idx, state, result, state_manager,
                    target_speaker, replacement_voice, progress_callback
                )
                
                if not success:
                    error_msg = f"Stage {stage_name} failed after {self.max_retries} retries"
                    logger.error(error_msg)
                    await save_failure_report(state, error_msg, job_checkpoint_dir)
                    raise RuntimeError(error_msg)
                
                # Update state after successful stage
                state.mark_stage_completed(stage_name)
                state_manager.save_state(state.to_dict())
            
            # Save final result
            await save_final_result(result, output_path)
            
            if progress_callback:
                progress_callback("Processing complete", len(PIPELINE_STAGES), len(PIPELINE_STAGES))
            
            logger.info(f"Pipeline job {job_id} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Pipeline job {job_id} failed: {e}")
            await save_failure_report(state, str(e), job_checkpoint_dir)
            raise
    
    def _generate_job_id(self, input_path: Union[str, Path]) -> str:
        """Generate a unique job ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        input_name = Path(input_path).stem
        return f"{input_name}_{timestamp}"
    
    async def _execute_stage_with_retry(
        self,
        stage_name: str,
        stage_idx: int,
        state: PipelineState,
        result: PipelineResult,
        state_manager: StateManager,
        target_speaker: Optional[str],
        replacement_voice: Optional[Union[str, Path]],
        progress_callback: Optional[Callable[[str, int, int], None]]
    ) -> bool:
        """Execute a pipeline stage with retry logic."""
        stage = self.stages[stage_name]
        
        for attempt in range(self.max_retries + 1):
            metrics = StageMetrics(
                stage_name=stage_name,
                status=StageStatus.RUNNING,
                start_time=datetime.now(timezone.utc),
                retry_count=attempt
            )
            
            try:
                if progress_callback:
                    progress_callback(f"Running {stage_name} (attempt {attempt + 1})", stage_idx, len(PIPELINE_STAGES))
                
                logger.info(f"Executing stage {stage_name} (attempt {attempt + 1}/{self.max_retries + 1})")
                
                # Record memory before stage
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                
                # Execute the stage
                stage_output = await self._execute_single_stage(
                    stage_name, stage, result, target_speaker, replacement_voice
                )
                
                # Save stage output to checkpoint
                checkpoint_path = result.save_stage_data(stage_name, stage_output)
                
                # Record metrics
                metrics.end_time = datetime.now(timezone.utc)
                metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
                metrics.status = StageStatus.COMPLETED
                
                if torch.cuda.is_available():
                    metrics.memory_peak_gb = torch.cuda.max_memory_allocated() / 1e9
                
                # Calculate checkpoint size
                if checkpoint_path.exists():
                    metrics.checkpoint_size_mb = checkpoint_path.stat().st_size / 1e6
                
                state.add_stage_metrics(metrics)
                state_manager.save_state(state.to_dict())
                logger.info(f"Stage {stage_name} completed successfully in {metrics.duration_seconds:.2f}s")
                return True
                
            except Exception as e:
                metrics.end_time = datetime.now(timezone.utc)
                metrics.duration_seconds = (metrics.end_time - metrics.start_time).total_seconds()
                metrics.status = StageStatus.FAILED
                metrics.error_message = str(e)
                metrics.error_type = type(e).__name__
                
                state.add_stage_metrics(metrics)
                state_manager.save_state(state.to_dict())
                
                logger.error(f"Stage {stage_name} failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.max_retries:
                    logger.info(f"Retrying stage {stage_name} in {DEFAULT_RETRY_DELAY_SECONDS} seconds...")
                    await asyncio.sleep(DEFAULT_RETRY_DELAY_SECONDS)
                    
                    # Clear GPU memory before retry
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                else:
                    logger.error(f"Stage {stage_name} failed after {self.max_retries} retries")
                    return False
        
        return False
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get detailed status of a pipeline job.
        
        Args:
            job_id: Job identifier
            
        Returns:
            Job status information
        """
        from .analytics_simple import PipelineAnalytics
        analytics = PipelineAnalytics(self.checkpoint_dir)
        return analytics.get_job_status(job_id)
    
    def list_jobs(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """List all pipeline jobs.
        
        Args:
            limit: Maximum number of jobs to return
            
        Returns:
            List of job summaries
        """
        from .analytics_simple import PipelineAnalytics
        analytics = PipelineAnalytics(self.checkpoint_dir)
        return analytics.list_jobs(limit)
    
    def get_stage_info(self) -> Dict[str, Dict[str, Any]]:
        """Get information about all pipeline stages.
        
        Returns:
            Stage information
        """
        return get_stage_info(self.stages)
    
    async def cleanup_job(self, job_id: str):
        """Clean up job checkpoints and temporary files.
        
        Args:
            job_id: Job identifier
        """
        job_dir = self.checkpoint_dir / job_id
        
        if job_dir.exists():
            import shutil
            shutil.rmtree(job_dir)
            logger.info(f"Cleaned up job {job_id}")
        else:
            logger.warning(f"Job directory not found: {job_dir}")
    
    async def cleanup(self):
        """Clean up pipeline resources."""
        await cleanup_pipeline_resources(self.stages)
    
    async def _execute_single_stage(
        self,
        stage_name: str,
        stage: Any,
        result: PipelineResult,
        target_speaker: Optional[str],
        replacement_voice: Optional[Union[str, Path]]
    ) -> Any:
        """Execute a single pipeline stage."""
        if stage_name == "preprocessor":
            return await stage.process(result.job_id)
        elif stage_name == "separator":
            if result.original_audio is None:
                result.original_audio, result.sample_rate = result.load_stage_data("preprocessor")
            return await stage.process(result.original_audio, result.sample_rate)
        elif stage_name == "diarizer":
            if not result.separated_audio:
                result.separated_audio = result.load_stage_data("separator")
            vocals = result.separated_audio.get('vocals', result.original_audio)
            return await stage.process(vocals, result.sample_rate)
        elif stage_name == "recognizer":
            if not result.speaker_segments:
                result.speaker_segments = result.load_stage_data("diarizer")
            vocals = result.separated_audio.get('vocals', result.original_audio)
            return await stage.process(vocals, result.sample_rate, result.speaker_segments)
        elif stage_name == "synthesizer":
            if not result.transcription:
                result.transcription = result.load_stage_data("recognizer")
            return await stage.process(result.transcription, target_speaker, replacement_voice)
        elif stage_name == "reconstructor":
            if not result.synthesized_audio:
                result.synthesized_audio = result.load_stage_data("synthesizer")
            return await stage.process(
                result.original_audio, result.separated_audio, 
                result.synthesized_audio, result.speaker_segments, result.sample_rate
            )

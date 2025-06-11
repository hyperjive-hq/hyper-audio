"""Helper functions for the core pipeline."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime, timezone

from ..utils.logging_utils import get_logger
from .constants import FAILURE_REPORT_FILENAME, PIPELINE_STAGES
from .models import PipelineState
from .checkpoint import StateManager

logger = get_logger("pipeline.helpers")


async def load_or_create_state(
    job_id: str,
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    state_manager: StateManager,
    resume_from_checkpoint: bool,
    config: Dict[str, Any]
) -> PipelineState:
    """Load existing pipeline state or create new one.
    
    Args:
        job_id: Job identifier
        input_path: Input file path
        output_path: Output file path
        state_manager: State manager instance
        resume_from_checkpoint: Whether to attempt resume
        config: Pipeline configuration
        
    Returns:
        Pipeline state
    """
    if resume_from_checkpoint and state_manager.state_exists():
        try:
            state_data = state_manager.load_state()
            if state_data:
                state = PipelineState.from_dict(state_data)
                logger.info(f"Resumed pipeline state for job {job_id}")
                return state
        except Exception as e:
            logger.warning(f"Failed to load checkpoint state: {e}")
            logger.info("Creating new pipeline state")
    
    # Create new state
    return PipelineState(
        job_id=job_id,
        input_path=str(input_path),
        output_path=str(output_path),
        config=config
    )


async def save_failure_report(
    state: PipelineState,
    error_message: str,
    checkpoint_dir: Path
):
    """Save detailed failure report.
    
    Args:
        state: Pipeline state
        error_message: Error description
        checkpoint_dir: Directory for saving report
    """
    import torch
    
    report_path = checkpoint_dir / FAILURE_REPORT_FILENAME
    
    report = {
        "job_id": state.job_id,
        "failure_time": datetime.now(timezone.utc).isoformat(),
        "error_message": error_message,
        "current_stage": state.current_stage,
        "stages_completed": state.stages_completed,
        "stage_metrics": [metric.to_dict() for metric in state.stage_metrics],
        "system_info": {
            "cuda_available": torch.cuda.is_available(),
            "memory_allocated_gb": torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else None,
            "memory_reserved_gb": torch.cuda.memory_reserved() / 1e9 if torch.cuda.is_available() else None
        }
    }
    
    try:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Failure report saved: {report_path}")
    except Exception as e:
        logger.error(f"Failed to save failure report: {e}")


async def save_final_result(result, output_path: Union[str, Path]):
    """Save the final processed audio.
    
    Args:
        result: Pipeline result with final audio
        output_path: Output file path
    """
    try:
        final_audio = result.final_audio
        sample_rate = result.sample_rate
        
        if final_audio is not None and sample_rate is not None:
            from ..utils.audio_utils import save_audio
            save_audio(final_audio, output_path, sample_rate)
            logger.info(f"Final audio saved to: {output_path}")
        else:
            logger.warning("No final audio to save")
    except Exception as e:
        logger.error(f"Failed to save final result: {e}")
        raise


def get_memory_usage() -> Dict[str, float]:
    """Get current GPU memory usage.
    
    Returns:
        Memory usage statistics
    """
    import torch
    
    if torch.cuda.is_available():
        return {
            "allocated_gb": torch.cuda.memory_allocated() / 1e9,
            "reserved_gb": torch.cuda.memory_reserved() / 1e9,
            "max_allocated_gb": torch.cuda.max_memory_allocated() / 1e9
        }
    return {"message": "CUDA not available"}


def get_stage_info(stages: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Get information about all pipeline stages.
    
    Args:
        stages: Dictionary of initialized stages
        
    Returns:
        Stage information
    """
    info = {}
    memory_info = get_memory_usage()
    
    for stage_def in PIPELINE_STAGES:
        stage_name = stage_def["name"]
        stage = stages.get(stage_name)
        
        info[stage_name] = {
            "name": stage_name,
            "description": stage_def["description"],
            "class": stage_def["class"],
            "status": "initialized" if stage else "not_loaded",
            "memory_usage": memory_info if stage else None
        }
    
    return info


async def cleanup_pipeline_resources(stages: Dict[str, Any]):
    """Clean up pipeline resources and models.
    
    Args:
        stages: Dictionary of pipeline stages
    """
    import torch
    
    logger.info("Cleaning up pipeline resources")
    
    # Cleanup individual stages
    for stage in stages.values():
        if hasattr(stage, 'cleanup'):
            try:
                await stage.cleanup()
            except Exception as e:
                logger.warning(f"Failed to cleanup stage {stage.__class__.__name__}: {e}")
    
    # Clear GPU memory
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU memory cleared")
    
    logger.info("Pipeline cleanup completed")


def validate_input_path(input_path: Union[str, Path]) -> Path:
    """Validate and convert input path.
    
    Args:
        input_path: Input file path
        
    Returns:
        Validated Path object
        
    Raises:
        FileNotFoundError: If input file doesn't exist
        ValueError: If input file is not a valid audio file
    """
    input_path = Path(input_path)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")
    
    # Check for common audio extensions
    audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.aac', '.ogg', '.wma'}
    if input_path.suffix.lower() not in audio_extensions:
        logger.warning(f"Input file extension '{input_path.suffix}' may not be a supported audio format")
    
    return input_path


def validate_output_path(output_path: Union[str, Path]) -> Path:
    """Validate and prepare output path.
    
    Args:
        output_path: Output file path
        
    Returns:
        Validated Path object
    """
    output_path = Path(output_path)
    
    # Create parent directory if it doesn't exist
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Warn if file exists
    if output_path.exists():
        logger.warning(f"Output file will be overwritten: {output_path}")
    
    return output_path
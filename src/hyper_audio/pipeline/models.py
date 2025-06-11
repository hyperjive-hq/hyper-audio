"""Data models for pipeline state and metrics."""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Any
from datetime import datetime, timezone
from pathlib import Path

from .constants import StageStatus, JobStatus


@dataclass
class StageMetrics:
    """Metrics for a pipeline stage execution."""
    stage_name: str
    status: StageStatus
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    duration_seconds: Optional[float] = None
    memory_peak_gb: Optional[float] = None
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    retry_count: int = 0
    checkpoint_size_mb: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable values."""
        data = asdict(self)
        data['status'] = self.status.value
        if self.start_time:
            data['start_time'] = self.start_time.isoformat()
        if self.end_time:
            data['end_time'] = self.end_time.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'StageMetrics':
        """Create from dictionary."""
        # Convert status back to enum
        if 'status' in data:
            data['status'] = StageStatus(data['status'])
        
        # Convert datetime strings back to datetime objects
        for field_name in ['start_time', 'end_time']:
            if data.get(field_name):
                data[field_name] = datetime.fromisoformat(data[field_name])
        
        return cls(**data)


@dataclass
class PipelineState:
    """Complete pipeline state for persistence."""
    job_id: str
    input_path: str
    output_path: str
    config: Dict[str, Any]
    current_stage: int = 0
    stages_completed: List[str] = field(default_factory=list)
    stage_metrics: List[StageMetrics] = field(default_factory=list)
    data_checksums: Dict[str, str] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with serializable values."""
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['stage_metrics'] = [metric.to_dict() for metric in self.stage_metrics]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineState':
        """Create from dictionary."""
        # Convert datetime strings back to datetime objects
        data['created_at'] = datetime.fromisoformat(data['created_at'])
        data['updated_at'] = datetime.fromisoformat(data['updated_at'])
        
        # Convert stage metrics
        stage_metrics = []
        for metric_dict in data.get('stage_metrics', []):
            stage_metrics.append(StageMetrics.from_dict(metric_dict))
        data['stage_metrics'] = stage_metrics
        
        return cls(**data)
    
    def update_timestamp(self):
        """Update the last modified timestamp."""
        self.updated_at = datetime.now(timezone.utc)
    
    def add_stage_metrics(self, metrics: StageMetrics):
        """Add stage metrics to the state."""
        self.stage_metrics.append(metrics)
        self.update_timestamp()
    
    def mark_stage_completed(self, stage_name: str):
        """Mark a stage as completed."""
        if stage_name not in self.stages_completed:
            self.stages_completed.append(stage_name)
        self.current_stage += 1
        self.update_timestamp()
    
    def get_stage_status(self, stage_name: str) -> StageStatus:
        """Get the status of a specific stage."""
        for metrics in self.stage_metrics:
            if metrics.stage_name == stage_name:
                return metrics.status
        return StageStatus.PENDING
    
    def get_overall_status(self, total_stages: int) -> JobStatus:
        """Determine overall job status."""
        if self.current_stage >= total_stages:
            return JobStatus.COMPLETED
        elif any(m.status == StageStatus.FAILED for m in self.stage_metrics):
            return JobStatus.FAILED
        else:
            return JobStatus.IN_PROGRESS


@dataclass
class JobSummary:
    """Summary information for a pipeline job."""
    job_id: str
    status: JobStatus
    progress_percentage: float
    created_at: datetime
    updated_at: datetime
    current_stage: int
    total_stages: int
    stages_completed: List[str]
    error_message: Optional[str] = None
    
    @classmethod
    def from_state(cls, state: PipelineState, total_stages: int) -> 'JobSummary':
        """Create job summary from pipeline state."""
        status = state.get_overall_status(total_stages)
        progress = (len(state.stages_completed) / total_stages) * 100 if total_stages > 0 else 0
        
        # Get latest error message if any
        error_message = None
        failed_metrics = [m for m in state.stage_metrics if m.status == StageStatus.FAILED]
        if failed_metrics:
            error_message = failed_metrics[-1].error_message
        
        return cls(
            job_id=state.job_id,
            status=status,
            progress_percentage=progress,
            created_at=state.created_at,
            updated_at=state.updated_at,
            current_stage=state.current_stage,
            total_stages=total_stages,
            stages_completed=state.stages_completed.copy(),
            error_message=error_message
        )


@dataclass
class PipelineResult:
    """Container for pipeline execution results."""
    job_id: str
    checkpoint_manager: Any  # CheckpointManager instance
    
    # Processing results - loaded lazily from checkpoints
    _original_audio: Optional[Any] = None
    _sample_rate: Optional[int] = None
    _separated_audio: Optional[Dict[str, Any]] = None
    _speaker_segments: Optional[List[Dict[str, Any]]] = None
    _transcription: Optional[Dict[str, Any]] = None
    _synthesized_audio: Optional[Dict[str, Any]] = None
    _final_audio: Optional[Any] = None
    
    # Metadata
    processing_stats: Dict[str, float] = field(default_factory=dict)
    stage_outputs: Dict[str, Path] = field(default_factory=dict)
    
    def get_stage_data(self, stage_name: str) -> Any:
        """Get data for a specific stage, loading from checkpoint if needed."""
        if stage_name == "preprocessor":
            if self._original_audio is None:
                data = self.checkpoint_manager.load_stage_data(stage_name)
                self._original_audio, self._sample_rate = data
            return self._original_audio, self._sample_rate
        
        elif stage_name == "separator":
            if self._separated_audio is None:
                self._separated_audio = self.checkpoint_manager.load_stage_data(stage_name)
            return self._separated_audio
        
        elif stage_name == "diarizer":
            if self._speaker_segments is None:
                self._speaker_segments = self.checkpoint_manager.load_stage_data(stage_name)
            return self._speaker_segments
        
        elif stage_name == "recognizer":
            if self._transcription is None:
                self._transcription = self.checkpoint_manager.load_stage_data(stage_name)
            return self._transcription
        
        elif stage_name == "synthesizer":
            if self._synthesized_audio is None:
                self._synthesized_audio = self.checkpoint_manager.load_stage_data(stage_name)
            return self._synthesized_audio
        
        elif stage_name == "reconstructor":
            if self._final_audio is None:
                self._final_audio = self.checkpoint_manager.load_stage_data(stage_name)
            return self._final_audio
        
        else:
            raise ValueError(f"Unknown stage: {stage_name}")
    
    def save_stage_data(self, stage_name: str, data: Any) -> Path:
        """Save stage data and update internal state."""
        checkpoint_path = self.checkpoint_manager.save_stage_data(stage_name, data)
        self.stage_outputs[stage_name] = checkpoint_path
        
        # Update internal cache
        if stage_name == "preprocessor":
            self._original_audio, self._sample_rate = data
        elif stage_name == "separator":
            self._separated_audio = data
        elif stage_name == "diarizer":
            self._speaker_segments = data
        elif stage_name == "recognizer":
            self._transcription = data
        elif stage_name == "synthesizer":
            self._synthesized_audio = data
        elif stage_name == "reconstructor":
            self._final_audio = data
        
        return checkpoint_path
    
    @property
    def original_audio(self):
        """Get original audio, loading from checkpoint if needed."""
        if self._original_audio is None:
            self.get_stage_data("preprocessor")
        return self._original_audio
    
    @property
    def sample_rate(self):
        """Get sample rate, loading from checkpoint if needed."""
        if self._sample_rate is None:
            self.get_stage_data("preprocessor")
        return self._sample_rate
    
    @property
    def separated_audio(self):
        """Get separated audio, loading from checkpoint if needed."""
        if self._separated_audio is None:
            self.get_stage_data("separator")
        return self._separated_audio
    
    @property
    def speaker_segments(self):
        """Get speaker segments, loading from checkpoint if needed."""
        if self._speaker_segments is None:
            self.get_stage_data("diarizer")
        return self._speaker_segments
    
    @property
    def transcription(self):
        """Get transcription, loading from checkpoint if needed."""
        if self._transcription is None:
            self.get_stage_data("recognizer")
        return self._transcription
    
    @property
    def synthesized_audio(self):
        """Get synthesized audio, loading from checkpoint if needed."""
        if self._synthesized_audio is None:
            self.get_stage_data("synthesizer")
        return self._synthesized_audio
    
    @property
    def final_audio(self):
        """Get final audio, loading from checkpoint if needed."""
        if self._final_audio is None:
            self.get_stage_data("reconstructor")
        return self._final_audio
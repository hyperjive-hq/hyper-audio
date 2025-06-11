"""Checkpoint management for pipeline resilience."""

import json
import pickle
import hashlib
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime, timezone

from ..utils.logging_utils import get_logger
from .constants import CHECKPOINT_EXTENSION, CHECKSUM_EXTENSION, STATE_FILENAME

logger = get_logger("pipeline.checkpoint")


class CheckpointManager:
    """Manages checkpointing for pipeline stages."""
    
    def __init__(self, checkpoint_dir: Path):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for storing checkpoints
        """
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def save_stage_data(self, stage_name: str, data: Any) -> Path:
        """Save stage output data to checkpoint.
        
        Args:
            stage_name: Name of the pipeline stage
            data: Data to checkpoint
            
        Returns:
            Path to the saved checkpoint file
            
        Raises:
            RuntimeError: If checkpoint saving fails
        """
        checkpoint_path = self.checkpoint_dir / f"{stage_name}_output{CHECKPOINT_EXTENSION}"
        
        try:
            # Save the data
            with open(checkpoint_path, 'wb') as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Calculate and store checksum for integrity
            checksum = self._calculate_checksum(checkpoint_path)
            checksum_path = self.checkpoint_dir / f"{stage_name}_checksum{CHECKSUM_EXTENSION}"
            
            with open(checksum_path, 'w') as f:
                f.write(checksum)
            
            file_size_mb = checkpoint_path.stat().st_size / 1e6
            logger.info(f"Saved {stage_name} checkpoint: {checkpoint_path} ({file_size_mb:.1f}MB)")
            
            return checkpoint_path
            
        except Exception as e:
            logger.error(f"Failed to save {stage_name} checkpoint: {e}")
            raise RuntimeError(f"Checkpoint save failed for {stage_name}: {e}") from e
    
    def load_stage_data(self, stage_name: str) -> Any:
        """Load stage output data from checkpoint.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            Loaded checkpoint data
            
        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is corrupted
            RuntimeError: If loading fails
        """
        checkpoint_path = self.checkpoint_dir / f"{stage_name}_output{CHECKPOINT_EXTENSION}"
        checksum_path = self.checkpoint_dir / f"{stage_name}_checksum{CHECKSUM_EXTENSION}"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        # Verify integrity if checksum exists
        if checksum_path.exists():
            self._verify_checkpoint_integrity(checkpoint_path, checksum_path, stage_name)
        
        try:
            with open(checkpoint_path, 'rb') as f:
                data = pickle.load(f)
            
            logger.info(f"Loaded {stage_name} checkpoint: {checkpoint_path}")
            return data
            
        except Exception as e:
            logger.error(f"Failed to load {stage_name} checkpoint: {e}")
            raise RuntimeError(f"Checkpoint load failed for {stage_name}: {e}") from e
    
    def checkpoint_exists(self, stage_name: str) -> bool:
        """Check if checkpoint exists for a stage.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            True if checkpoint exists, False otherwise
        """
        checkpoint_path = self.checkpoint_dir / f"{stage_name}_output{CHECKPOINT_EXTENSION}"
        return checkpoint_path.exists()
    
    def get_checkpoint_info(self, stage_name: str) -> Optional[Dict[str, Any]]:
        """Get information about a checkpoint.
        
        Args:
            stage_name: Name of the pipeline stage
            
        Returns:
            Checkpoint information or None if not found
        """
        checkpoint_path = self.checkpoint_dir / f"{stage_name}_output{CHECKPOINT_EXTENSION}"
        
        if not checkpoint_path.exists():
            return None
        
        stat = checkpoint_path.stat()
        return {
            "stage_name": stage_name,
            "file_path": str(checkpoint_path),
            "size_mb": stat.st_size / 1e6,
            "created_at": datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc),
            "modified_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
        }
    
    def cleanup_stage_checkpoints(self, stage_name: str):
        """Remove checkpoints for a specific stage.
        
        Args:
            stage_name: Name of the pipeline stage
        """
        checkpoint_path = self.checkpoint_dir / f"{stage_name}_output{CHECKPOINT_EXTENSION}"
        checksum_path = self.checkpoint_dir / f"{stage_name}_checksum{CHECKSUM_EXTENSION}"
        
        for path in [checkpoint_path, checksum_path]:
            if path.exists():
                path.unlink()
                logger.debug(f"Removed checkpoint file: {path}")
    
    def cleanup_all_checkpoints(self):
        """Remove all checkpoint files."""
        for file_path in self.checkpoint_dir.glob("*"):
            if file_path.is_file():
                file_path.unlink()
                logger.debug(f"Removed checkpoint file: {file_path}")
    
    def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate SHA256 checksum of a file.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Hexadecimal checksum string
        """
        hash_sha256 = hashlib.sha256()
        
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        
        return hash_sha256.hexdigest()
    
    def _verify_checkpoint_integrity(self, checkpoint_path: Path, checksum_path: Path, stage_name: str):
        """Verify checkpoint file integrity.
        
        Args:
            checkpoint_path: Path to checkpoint file
            checksum_path: Path to checksum file
            stage_name: Name of the stage (for error reporting)
            
        Raises:
            ValueError: If checkpoint is corrupted
        """
        try:
            with open(checksum_path, 'r') as f:
                expected_checksum = f.read().strip()
            
            actual_checksum = self._calculate_checksum(checkpoint_path)
            
            if actual_checksum != expected_checksum:
                raise ValueError(f"Checkpoint corruption detected for {stage_name}")
                
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            raise ValueError(f"Failed to verify checkpoint integrity for {stage_name}: {e}") from e


class StateManager:
    """Manages pipeline state persistence."""
    
    def __init__(self, checkpoint_dir: Path):
        """Initialize state manager.
        
        Args:
            checkpoint_dir: Directory for storing state
        """
        self.checkpoint_dir = checkpoint_dir
        self.state_path = checkpoint_dir / STATE_FILENAME
    
    def save_state(self, state: Dict[str, Any]):
        """Save pipeline state to disk.
        
        Args:
            state: Pipeline state dictionary
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(self.state_path, 'w') as f:
                json.dump(state, f, indent=2, default=self._json_serializer)
            
            logger.debug(f"Pipeline state saved: {self.state_path}")
            
        except Exception as e:
            logger.error(f"Failed to save pipeline state: {e}")
            raise RuntimeError(f"State save failed: {e}") from e
    
    def load_state(self) -> Optional[Dict[str, Any]]:
        """Load pipeline state from disk.
        
        Returns:
            Pipeline state dictionary or None if not found
        """
        if not self.state_path.exists():
            return None
        
        try:
            with open(self.state_path, 'r') as f:
                state = json.load(f)
            
            logger.debug(f"Pipeline state loaded: {self.state_path}")
            return state
            
        except Exception as e:
            logger.error(f"Failed to load pipeline state: {e}")
            raise RuntimeError(f"State load failed: {e}") from e
    
    def state_exists(self) -> bool:
        """Check if state file exists.
        
        Returns:
            True if state exists, False otherwise
        """
        return self.state_path.exists()
    
    def _json_serializer(self, obj):
        """Custom JSON serializer for datetime objects."""
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
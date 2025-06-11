"""Tests for checkpoint management functionality."""

import pytest
import pickle
import hashlib
from pathlib import Path
from unittest.mock import patch, mock_open

from src.hyper_audio.pipeline.checkpoint import CheckpointManager, StateManager
from src.hyper_audio.pipeline.constants import CHECKPOINT_EXTENSION, CHECKSUM_EXTENSION


class TestCheckpointManager:
    """Test checkpoint management functionality."""
    
    def test_init(self, checkpoint_manager, checkpoint_dir, job_id):
        """Test checkpoint manager initialization."""
        assert checkpoint_manager.checkpoint_dir == checkpoint_dir / job_id
        assert checkpoint_manager.checkpoint_dir.exists()
    
    def test_save_stage_data(self, checkpoint_manager, sample_audio_data):
        """Test saving stage data to checkpoint."""
        stage_name = "test_stage"
        test_data = {"audio": sample_audio_data[0], "sample_rate": sample_audio_data[1]}
        
        # Save the data
        checkpoint_path = checkpoint_manager.save_stage_data(stage_name, test_data)
        
        # Verify files were created
        expected_checkpoint = checkpoint_manager.checkpoint_dir / f"{stage_name}_output{CHECKPOINT_EXTENSION}"
        expected_checksum = checkpoint_manager.checkpoint_dir / f"{stage_name}_checksum{CHECKSUM_EXTENSION}"
        
        assert checkpoint_path == expected_checkpoint
        assert expected_checkpoint.exists()
        assert expected_checksum.exists()
        
        # Verify data can be loaded
        with open(expected_checkpoint, 'rb') as f:
            loaded_data = pickle.load(f)
        
        assert "audio" in loaded_data
        assert "sample_rate" in loaded_data
        assert loaded_data["sample_rate"] == test_data["sample_rate"]
    
    def test_load_stage_data(self, checkpoint_manager, sample_audio_data):
        """Test loading stage data from checkpoint."""
        stage_name = "test_stage"
        test_data = {"audio": sample_audio_data[0], "sample_rate": sample_audio_data[1]}
        
        # Save data first
        checkpoint_manager.save_stage_data(stage_name, test_data)
        
        # Load the data
        loaded_data = checkpoint_manager.load_stage_data(stage_name)
        
        assert "audio" in loaded_data
        assert "sample_rate" in loaded_data
        assert loaded_data["sample_rate"] == test_data["sample_rate"]
    
    def test_load_nonexistent_checkpoint(self, checkpoint_manager):
        """Test loading from non-existent checkpoint raises error."""
        with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
            checkpoint_manager.load_stage_data("nonexistent_stage")
    
    def test_checksum_verification(self, checkpoint_manager, sample_audio_data):
        """Test checksum verification prevents corruption."""
        stage_name = "test_stage"
        test_data = {"audio": sample_audio_data[0], "sample_rate": sample_audio_data[1]}
        
        # Save data
        checkpoint_manager.save_stage_data(stage_name, test_data)
        
        # Corrupt the checkpoint file
        checkpoint_path = checkpoint_manager.checkpoint_dir / f"{stage_name}_output{CHECKPOINT_EXTENSION}"
        with open(checkpoint_path, 'wb') as f:
            f.write(b"corrupted data")
        
        # Loading should detect corruption
        with pytest.raises(ValueError, match="Checkpoint corruption detected"):
            checkpoint_manager.load_stage_data(stage_name)
    
    def test_checkpoint_exists(self, checkpoint_manager, sample_audio_data):
        """Test checking if checkpoint exists."""
        stage_name = "test_stage"
        
        # Initially should not exist
        assert not checkpoint_manager.checkpoint_exists(stage_name)
        
        # After saving should exist
        checkpoint_manager.save_stage_data(stage_name, sample_audio_data)
        assert checkpoint_manager.checkpoint_exists(stage_name)
    
    def test_get_checkpoint_info(self, checkpoint_manager, sample_audio_data):
        """Test getting checkpoint information."""
        stage_name = "test_stage"
        
        # No info for non-existent checkpoint
        assert checkpoint_manager.get_checkpoint_info(stage_name) is None
        
        # Save data and get info
        checkpoint_manager.save_stage_data(stage_name, sample_audio_data)
        info = checkpoint_manager.get_checkpoint_info(stage_name)
        
        assert info is not None
        assert info["stage_name"] == stage_name
        assert "file_path" in info
        assert "size_mb" in info
        assert "created_at" in info
        assert "modified_at" in info
        assert info["size_mb"] > 0
    
    def test_cleanup_stage_checkpoints(self, checkpoint_manager, sample_audio_data):
        """Test cleaning up stage checkpoints."""
        stage_name = "test_stage"
        
        # Save data
        checkpoint_manager.save_stage_data(stage_name, sample_audio_data)
        
        # Verify files exist
        assert checkpoint_manager.checkpoint_exists(stage_name)
        
        # Cleanup
        checkpoint_manager.cleanup_stage_checkpoints(stage_name)
        
        # Verify files are gone
        assert not checkpoint_manager.checkpoint_exists(stage_name)
    
    def test_cleanup_all_checkpoints(self, checkpoint_manager, sample_audio_data):
        """Test cleaning up all checkpoints."""
        # Save multiple checkpoints
        for stage_name in ["stage1", "stage2", "stage3"]:
            checkpoint_manager.save_stage_data(stage_name, sample_audio_data)
        
        # Verify files exist
        assert len(list(checkpoint_manager.checkpoint_dir.glob("*"))) > 0
        
        # Cleanup all
        checkpoint_manager.cleanup_all_checkpoints()
        
        # Verify directory is empty
        assert len(list(checkpoint_manager.checkpoint_dir.glob("*"))) == 0
    
    def test_save_large_data(self, checkpoint_manager):
        """Test saving large data objects."""
        stage_name = "large_stage"
        # Create a large data structure
        large_data = {
            "large_array": list(range(100000)),
            "metadata": {"size": "large", "type": "test"}
        }
        
        # Should handle large data without issues
        checkpoint_path = checkpoint_manager.save_stage_data(stage_name, large_data)
        assert checkpoint_path.exists()
        
        # Verify we can load it back
        loaded_data = checkpoint_manager.load_stage_data(stage_name)
        assert len(loaded_data["large_array"]) == 100000
        assert loaded_data["metadata"]["size"] == "large"
    
    def test_save_error_handling(self, checkpoint_manager):
        """Test error handling during save operations."""
        stage_name = "error_stage"
        test_data = {"test": "data"}
        
        # Mock open to raise an exception
        with patch("builtins.open", side_effect=IOError("Disk full")):
            with pytest.raises(RuntimeError, match="Checkpoint save failed"):
                checkpoint_manager.save_stage_data(stage_name, test_data)
    
    def test_load_error_handling(self, checkpoint_manager, sample_audio_data):
        """Test error handling during load operations."""
        stage_name = "error_stage"
        
        # Save valid data first
        checkpoint_manager.save_stage_data(stage_name, sample_audio_data)
        
        # Mock pickle.load to raise an exception
        with patch("pickle.load", side_effect=pickle.PickleError("Corrupt pickle")):
            with pytest.raises(RuntimeError, match="Checkpoint load failed"):
                checkpoint_manager.load_stage_data(stage_name)


class TestStateManager:
    """Test state management functionality."""
    
    def test_init(self, state_manager, checkpoint_dir, job_id):
        """Test state manager initialization."""
        assert state_manager.checkpoint_dir == checkpoint_dir / job_id
        assert state_manager.state_path.name == "pipeline_state.json"
    
    def test_save_and_load_state(self, state_manager, sample_pipeline_state):
        """Test saving and loading pipeline state."""
        # Save state
        state_dict = sample_pipeline_state.to_dict()
        state_manager.save_state(state_dict)
        
        # Verify file exists
        assert state_manager.state_exists()
        
        # Load state
        loaded_state = state_manager.load_state()
        
        assert loaded_state is not None
        assert loaded_state["job_id"] == sample_pipeline_state.job_id
        assert loaded_state["current_stage"] == sample_pipeline_state.current_stage
        assert len(loaded_state["stage_metrics"]) == len(sample_pipeline_state.stage_metrics)
    
    def test_load_nonexistent_state(self, state_manager):
        """Test loading non-existent state returns None."""
        assert not state_manager.state_exists()
        assert state_manager.load_state() is None
    
    def test_state_exists(self, state_manager, sample_pipeline_state):
        """Test checking if state exists."""
        # Initially should not exist
        assert not state_manager.state_exists()
        
        # After saving should exist
        state_manager.save_state(sample_pipeline_state.to_dict())
        assert state_manager.state_exists()
    
    def test_save_state_creates_directory(self, temp_dir, job_id, sample_pipeline_state):
        """Test that save_state creates directory if it doesn't exist."""
        # Create state manager with non-existent directory
        non_existent_dir = temp_dir / "new_checkpoints" / job_id
        state_manager = StateManager(non_existent_dir)
        
        # Directory should not exist initially
        assert not non_existent_dir.exists()
        
        # Save state should create directory
        state_manager.save_state(sample_pipeline_state.to_dict())
        
        # Directory should now exist
        assert non_existent_dir.exists()
        assert state_manager.state_path.exists()
    
    def test_datetime_serialization(self, state_manager, sample_pipeline_state):
        """Test that datetime objects are properly serialized."""
        # Save state with datetime objects
        state_dict = sample_pipeline_state.to_dict()
        state_manager.save_state(state_dict)
        
        # Load and verify datetimes are strings
        loaded_state = state_manager.load_state()
        
        assert isinstance(loaded_state["created_at"], str)
        assert isinstance(loaded_state["updated_at"], str)
        
        # Verify we can parse them back to datetime
        from datetime import datetime
        datetime.fromisoformat(loaded_state["created_at"])
        datetime.fromisoformat(loaded_state["updated_at"])
    
    def test_save_error_handling(self, state_manager, sample_pipeline_state):
        """Test error handling during save operations."""
        with patch("builtins.open", side_effect=IOError("Permission denied")):
            with pytest.raises(RuntimeError, match="State save failed"):
                state_manager.save_state(sample_pipeline_state.to_dict())
    
    def test_load_error_handling(self, state_manager, sample_pipeline_state):
        """Test error handling during load operations."""
        # Save valid state first
        state_manager.save_state(sample_pipeline_state.to_dict())
        
        # Mock json.load to raise an exception
        with patch("json.load", side_effect=ValueError("Invalid JSON")):
            with pytest.raises(RuntimeError, match="State load failed"):
                state_manager.load_state()
    
    def test_custom_json_serializer(self, state_manager):
        """Test custom JSON serializer for datetime objects."""
        from datetime import datetime
        dt = datetime.now()
        
        # Test datetime serialization
        serialized = state_manager._json_serializer(dt)
        assert isinstance(serialized, str)
        assert dt.isoformat() == serialized
        
        # Test non-datetime object raises TypeError
        with pytest.raises(TypeError, match="Object.*is not JSON serializable"):
            state_manager._json_serializer(object())
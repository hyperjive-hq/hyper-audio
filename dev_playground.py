#!/usr/bin/env python3
"""Development playground for individual pipeline stage testing and refinement.

This script provides a comfortable environment for developing and testing
individual pipeline stages without running the full pipeline.
"""

import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple
import argparse
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hyper_audio.pipeline.stages import (
    AudioPreprocessor, VoiceSeparator, SpeakerDiarizer,
    SpeechRecognizer, VoiceSynthesizer, AudioReconstructor
)
from hyper_audio.pipeline.constants import PIPELINE_STAGES
from hyper_audio.config.settings import settings
from hyper_audio.utils.logging_utils import get_logger
from hyper_audio.utils.audio_utils import save_audio

logger = get_logger("dev_playground")


class StagePlayground:
    """Interactive development environment for pipeline stages."""
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "dev_cache"):
        """Initialize the playground.
        
        Args:
            data_dir: Directory containing sample data
            cache_dir: Directory for caching intermediate results
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize stages
        self.stages = {
            "preprocessor": AudioPreprocessor(),
            "separator": VoiceSeparator(),
            "diarizer": SpeakerDiarizer(),
            "recognizer": SpeechRecognizer(),
            "synthesizer": VoiceSynthesizer(),
            "reconstructor": AudioReconstructor()
        }
        
        # Cache for intermediate results
        self.cache = {}
        
        logger.info(f"StagePlayground initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Device: {settings.device}")
    
    def list_stages(self) -> List[str]:
        """List available pipeline stages."""
        return list(self.stages.keys())
    
    def list_sample_data(self) -> Dict[str, List[str]]:
        """List available sample data files."""
        sample_data = {}
        
        # Audio files
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        audio_files = []
        for ext in audio_extensions:
            audio_files.extend([f.name for f in self.data_dir.glob(f"**/*{ext}")])
        sample_data['audio'] = audio_files
        
        # Cache files (intermediate results)
        cache_files = [f.name for f in self.cache_dir.glob("*.pkl")]
        sample_data['cached_results'] = cache_files
        
        return sample_data
    
    async def run_stage(self, 
                       stage_name: str, 
                       input_data: Any = None,
                       input_file: Optional[str] = None,
                       cache_result: bool = True,
                       profile: bool = False,
                       **kwargs) -> Any:
        """Run a single pipeline stage.
        
        Args:
            stage_name: Name of the stage to run
            input_data: Direct input data (overrides input_file)
            input_file: Path to input file or cached result
            cache_result: Whether to cache the result
            profile: Whether to profile execution
            **kwargs: Additional arguments for the stage
            
        Returns:
            Stage output
        """
        if stage_name not in self.stages:
            raise ValueError(f"Unknown stage: {stage_name}. Available: {list(self.stages.keys())}")
        
        stage = self.stages[stage_name]
        
        # Prepare input
        if input_data is None:
            if input_file:
                input_data = self.load_data(input_file)
            else:
                # Try to find appropriate sample data
                input_data = self._get_default_input(stage_name)
        
        # Profile execution if requested
        start_time = time.time()
        memory_before = self._get_memory_usage()
        
        logger.info(f"Running stage: {stage_name}")
        logger.info(f"Input type: {type(input_data)}")
        
        try:
            # Run the stage
            if stage_name == "preprocessor":
                result = await stage.process(input_data, **kwargs)
            elif stage_name == "separator":
                if isinstance(input_data, tuple):
                    audio, sample_rate = input_data
                    result = await stage.process(audio, sample_rate, **kwargs)
                else:
                    raise ValueError(f"Separator stage expects (audio_data, sample_rate) tuple, got {type(input_data)}")
            elif stage_name == "diarizer":
                if isinstance(input_data, tuple):
                    audio, sample_rate = input_data
                    result = await stage.process(audio, sample_rate, **kwargs)
                elif isinstance(input_data, dict) and "vocals" in input_data:
                    # Input from separator stage
                    vocals = input_data["vocals"]
                    # Get sample rate from metadata or default
                    sample_rate = input_data.get("_sample_rate", kwargs.get("sample_rate", 44100))
                    result = await stage.process(vocals, sample_rate, **kwargs)
                else:
                    raise ValueError(f"Diarizer stage expects (vocals_audio, sample_rate) or separator output dict, got {type(input_data)}")
            elif stage_name == "recognizer":
                # Expects (audio, sample_rate, speaker_segments)
                if isinstance(input_data, tuple) and len(input_data) == 3:
                    audio, sample_rate, segments = input_data
                    result = await stage.process(audio, sample_rate, segments, **kwargs)
                else:
                    result = await stage.process(input_data, **kwargs)
            elif stage_name == "synthesizer":
                # Expects transcription and voice parameters
                result = await stage.process(input_data, **kwargs)
            elif stage_name == "reconstructor":
                # Expects multiple audio components
                result = await stage.process(input_data, **kwargs)
            else:
                result = await stage.process(input_data, **kwargs)
            
            # Profile results
            end_time = time.time()
            memory_after = self._get_memory_usage()
            
            if profile:
                self._print_profile_info(
                    stage_name, start_time, end_time, 
                    memory_before, memory_after, result
                )
            
            # Cache result if requested
            if cache_result:
                cache_key = f"{stage_name}_{int(time.time())}"
                # Extract sample rate from result for audio saving
                sample_rate = None
                if isinstance(result, tuple) and len(result) == 2:
                    # Preprocessor result: (audio, sample_rate)
                    _, sample_rate = result
                elif isinstance(input_data, tuple) and len(input_data) >= 2:
                    # Use input sample rate for other stages
                    sample_rate = input_data[1]
                
                self.save_result(cache_key, result, sample_rate)
                logger.info(f"Result cached as: {cache_key}")
            
            logger.info(f"Stage {stage_name} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            raise
    
    def save_result(self, name: str, data: Any, sample_rate: Optional[int] = None) -> Path:
        """Save result to cache with directory structure and audio files."""
        # Create result directory
        result_dir = self.cache_dir / name
        result_dir.mkdir(exist_ok=True)
        
        # Save pickle data
        pickle_path = result_dir / "data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Save metadata
        metadata = {
            "timestamp": time.time(),
            "type": str(type(data)),
            "stage": name.split('_')[0],
            "sample_rate": sample_rate
        }
        
        metadata_path = result_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save audio files if applicable
        self._save_audio_files(result_dir, data, sample_rate)
        
        logger.info(f"Saved result to: {result_dir}")
        return result_dir
    
    def _save_audio_files(self, result_dir: Path, data: Any, sample_rate: Optional[int]) -> None:
        """Save audio data as files for easy inspection."""
        try:
            # Handle different data types
            if isinstance(data, tuple) and len(data) == 2:
                # Preprocessor output: (audio_data, sample_rate)
                audio_data, sr = data
                if isinstance(audio_data, np.ndarray):
                    save_audio(audio_data, result_dir / "audio.wav", sr)
                    
            elif isinstance(data, dict):
                # Separator output: {"vocals": array, "music": array}
                # Synthesizer output: {"Speaker_A": array, "timing_info": [...]}
                for key, value in data.items():
                    if isinstance(value, np.ndarray) and value.ndim >= 1 and sample_rate is not None:
                        filename = f"{key}.wav"
                        save_audio(value, result_dir / filename, sample_rate)
                        
            elif isinstance(data, list):
                # Diarizer/Recognizer output: save as JSON for readability
                json_path = result_dir / "segments.json"
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                logger.info(f"Saved segments to: {json_path}")
                    
            elif isinstance(data, np.ndarray) and sample_rate is not None:
                # Raw audio array
                save_audio(data, result_dir / "audio.wav", sample_rate)
                
        except Exception as e:
            logger.warning(f"Could not save audio files: {e}")
    
    def load_data(self, filename: str) -> Any:
        """Load data from file (cache or sample data)."""
        # Convert to Path and resolve any relative paths
        input_path = Path(filename).resolve()
        
        # If the resolved path exists, use it directly
        if input_path.exists():
            return str(input_path)
        
        # Try cache directory structure first
        cache_dir_path = self.cache_dir / filename
        if cache_dir_path.is_dir():
            data_file = cache_dir_path / "data.pkl"
            metadata_file = cache_dir_path / "metadata.json"
            if data_file.exists():
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                
                # Enhance data with metadata if available
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    # Add sample rate to data for stages that need it
                    if isinstance(data, dict) and "sample_rate" in metadata:
                        data["_sample_rate"] = metadata["sample_rate"]
                
                return data
        
        # Try old pickle format for backward compatibility
        cache_path = self.cache_dir / filename
        if cache_path.exists() and cache_path.is_file():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Try adding .pkl extension for old format
        if not filename.endswith('.pkl'):
            cache_path = self.cache_dir / f"{filename}.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Try data directory (for sample files)
        data_path = self.data_dir / Path(filename).name
        if data_path.exists():
            return str(data_path)
        
        raise FileNotFoundError(f"Data file not found: {filename}")
    
    def _get_default_input(self, stage_name: str) -> Any:
        """Get default input for a stage from sample data or latest cache."""
        
        if stage_name == "preprocessor":
            # Need audio file from data directory
            sample_data = self.list_sample_data()
            if sample_data['audio']:
                selected_file = str(self.data_dir / sample_data['audio'][0])
                logger.info(f"Auto-selected audio file: {selected_file}")
                return selected_file
            else:
                raise ValueError("No sample audio files found for preprocessor")
        
        # For other stages, find the most recent output from the previous stage
        stage_order = ["preprocessor", "separator", "diarizer", "recognizer", "synthesizer", "reconstructor"]
        current_idx = stage_order.index(stage_name)
        
        if current_idx > 0:
            prev_stage = stage_order[current_idx - 1]
            latest_result = self._find_latest_stage_output(prev_stage)
            if latest_result:
                logger.info(f"Auto-selected latest {prev_stage} output: {latest_result}")
                return self.load_data(latest_result)
        
        raise ValueError(f"No suitable input found for stage: {stage_name}")
    
    def _find_latest_stage_output(self, stage_name: str) -> Optional[str]:
        """Find the most recent output from a specific stage."""
        # Look for directories in cache that start with stage name
        stage_dirs = []
        for item in self.cache_dir.iterdir():
            if item.is_dir() and item.name.startswith(f"{stage_name}_"):
                try:
                    # Extract timestamp from directory name
                    timestamp_str = item.name.split('_', 1)[1]
                    timestamp = int(timestamp_str)
                    stage_dirs.append((timestamp, item.name))
                except (ValueError, IndexError):
                    continue
        
        # Also look for old pickle files for backward compatibility
        for item in self.cache_dir.iterdir():
            if item.is_file() and item.name.startswith(f"{stage_name}_") and item.name.endswith('.pkl'):
                try:
                    # Extract timestamp from filename
                    timestamp_str = item.name.replace(f"{stage_name}_", "").replace(".pkl", "")
                    timestamp = int(timestamp_str)
                    stage_dirs.append((timestamp, item.name.replace(".pkl", "")))
                except (ValueError, IndexError):
                    continue
        
        if stage_dirs:
            # Sort by timestamp (most recent first) and return the latest
            stage_dirs.sort(reverse=True)
            latest = stage_dirs[0][1]
            return latest
        
        return None
    
    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage."""
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        
        result = {
            'ram_mb': memory_info.rss / 1024 / 1024,
            'gpu_mb': 0
        }
        
        try:
            import torch
            if torch.cuda.is_available():
                result['gpu_mb'] = torch.cuda.memory_allocated() / 1024 / 1024
        except ImportError:
            pass
        
        return result
    
    def _print_profile_info(self, stage_name: str, start_time: float, end_time: float,
                           memory_before: Dict, memory_after: Dict, result: Any):
        """Print profiling information."""
        duration = end_time - start_time
        ram_diff = memory_after['ram_mb'] - memory_before['ram_mb']
        gpu_diff = memory_after['gpu_mb'] - memory_before['gpu_mb']
        
        print(f"\n--- Performance Profile: {stage_name} ---")
        print(f"Execution time: {duration:.2f}s")
        print(f"RAM usage: {ram_diff:+.1f} MB (now: {memory_after['ram_mb']:.1f} MB)")
        print(f"GPU memory: {gpu_diff:+.1f} MB (now: {memory_after['gpu_mb']:.1f} MB)")
        print(f"Result type: {type(result)}")
        if hasattr(result, 'shape'):
            print(f"Result shape: {result.shape}")
        elif isinstance(result, (list, dict)):
            print(f"Result length: {len(result)}")
        print("-" * 40)
    
    def inspect_result(self, data: Any) -> Dict[str, Any]:
        """Inspect and summarize result data."""
        info = {
            'type': str(type(data)),
            'size': sys.getsizeof(data)
        }
        
        if hasattr(data, 'shape'):
            info['shape'] = data.shape
            info['dtype'] = str(data.dtype) if hasattr(data, 'dtype') else 'unknown'
        elif isinstance(data, (list, tuple)):
            info['length'] = len(data)
            if data:
                info['first_element_type'] = str(type(data[0]))
        elif isinstance(data, dict):
            info['keys'] = list(data.keys())
            info['length'] = len(data)
        
        return info


async def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(description="Pipeline Stage Development Playground")
    parser.add_argument("command", choices=["list", "run", "inspect", "profile"], 
                       help="Command to execute")
    parser.add_argument("--stage", "-s", help="Stage name to operate on")
    parser.add_argument("--input", "-i", help="Input file or cached result")
    parser.add_argument("--cache-name", "-c", help="Name for caching result")
    parser.add_argument("--data-dir", "-d", default="data", help="Data directory")
    parser.add_argument("--no-cache", action="store_true", help="Don't cache results")
    
    args = parser.parse_args()
    
    playground = StagePlayground(data_dir=args.data_dir)
    
    if args.command == "list":
        if args.stage:
            # List sample data
            sample_data = playground.list_sample_data()
            print(f"Available sample data:")
            for category, files in sample_data.items():
                print(f"  {category}: {files}")
        else:
            # List stages
            stages = playground.list_stages()
            print(f"Available stages: {stages}")
            
            # Show stage info
            print(f"\nStage descriptions:")
            for stage_def in PIPELINE_STAGES:
                print(f"  {stage_def['name']}: {stage_def['description']}")
    
    elif args.command == "run":
        if not args.stage:
            print("Error: --stage required for run command")
            return
        
        try:
            result = await playground.run_stage(
                args.stage,
                input_file=args.input,
                cache_result=not args.no_cache,
                profile=True
            )
            
            # Show result summary
            info = playground.inspect_result(result)
            print(f"\nResult summary: {info}")
            
        except Exception as e:
            print(f"Error running stage {args.stage}: {e}")
    
    elif args.command == "inspect":
        if not args.input:
            print("Error: --input required for inspect command")
            return
        
        try:
            data = playground.load_data(args.input)
            info = playground.inspect_result(data)
            print(f"Data inspection for '{args.input}':")
            for key, value in info.items():
                print(f"  {key}: {value}")
        except Exception as e:
            print(f"Error inspecting {args.input}: {e}")
    
    elif args.command == "profile":
        if not args.stage:
            print("Error: --stage required for profile command")
            return
        
        print(f"Profiling stage: {args.stage}")
        try:
            await playground.run_stage(
                args.stage,
                input_file=args.input,
                cache_result=not args.no_cache,
                profile=True
            )
        except Exception as e:
            print(f"Error profiling stage {args.stage}: {e}")


if __name__ == "__main__":
    asyncio.run(main())
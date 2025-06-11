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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hyper_audio.pipeline.stages import (
    AudioPreprocessor, VoiceSeparator, SpeakerDiarizer,
    SpeechRecognizer, VoiceSynthesizer, AudioReconstructor
)
from hyper_audio.pipeline.constants import PIPELINE_STAGES
from hyper_audio.config.settings import settings
from hyper_audio.utils.logging_utils import get_logger

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
                    result = await stage.process(input_data, **kwargs)
            elif stage_name == "diarizer":
                if isinstance(input_data, tuple):
                    audio, sample_rate = input_data
                    result = await stage.process(audio, sample_rate, **kwargs)
                else:
                    result = await stage.process(input_data, **kwargs)
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
                self.save_result(cache_key, result)
                logger.info(f"Result cached as: {cache_key}")
            
            logger.info(f"Stage {stage_name} completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            raise
    
    def save_result(self, name: str, data: Any) -> Path:
        """Save result to cache."""
        cache_path = self.cache_dir / f"{name}.pkl"
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
        logger.info(f"Saved result to: {cache_path}")
        return cache_path
    
    def load_data(self, filename: str) -> Any:
        """Load data from file (cache or sample data)."""
        # Try cache first
        cache_path = self.cache_dir / filename
        if cache_path.exists():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        # Try adding .pkl extension
        if not filename.endswith('.pkl'):
            cache_path = self.cache_dir / f"{filename}.pkl"
            if cache_path.exists():
                with open(cache_path, 'rb') as f:
                    return pickle.load(f)
        
        # Try data directory
        data_path = self.data_dir / filename
        if data_path.exists():
            return str(data_path)
        
        raise FileNotFoundError(f"Data file not found: {filename}")
    
    def _get_default_input(self, stage_name: str) -> Any:
        """Get default input for a stage from sample data."""
        sample_data = self.list_sample_data()
        
        if stage_name == "preprocessor":
            # Need audio file
            if sample_data['audio']:
                return str(self.data_dir / sample_data['audio'][0])
            else:
                raise ValueError("No sample audio files found for preprocessor")
        
        # For other stages, try to find cached results from previous stages
        stage_order = ["preprocessor", "separator", "diarizer", "recognizer", "synthesizer", "reconstructor"]
        current_idx = stage_order.index(stage_name)
        
        # Look for cached results from previous stage
        if current_idx > 0:
            prev_stage = stage_order[current_idx - 1]
            prev_results = [f for f in sample_data['cached_results'] if f.startswith(prev_stage)]
            if prev_results:
                return self.load_data(prev_results[0])
        
        raise ValueError(f"No suitable input found for stage: {stage_name}")
    
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
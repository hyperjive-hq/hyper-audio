#!/usr/bin/env python3
"""Enhanced development playground for pipeline stages with dependency resolution.

This script provides an intelligent development environment for testing pipeline stages
using the new configurable interface with automatic dependency resolution and user guidance.
"""

import asyncio
import json
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Tuple, Set
import argparse
import sys
import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hyper_audio.pipeline.stage_interface import EnhancedPipelineStage, DataType
from hyper_audio.pipeline.config_manager import ConfigManager
# Import all stages - fail fast if dependencies are missing
from hyper_audio.pipeline.stages import (
    AudioPreprocessor, VoiceSeparator, SpeakerDiarizer,
    SpeechRecognizer, VoiceSynthesizer, AudioReconstructor,
    SepformerSeparator, EnhancedVoiceSeparator, SpeechEnhancer
)
from hyper_audio.config.settings import settings
from hyper_audio.utils.logging_utils import get_logger
from hyper_audio.utils.audio_utils import save_audio

logger = get_logger("dev_playground_enhanced")


class DependencyResolver:
    """Resolves dependencies between pipeline stages."""
    
    def __init__(self, registered_stages: Dict[str, type]):
        """Initialize with registered stage classes."""
        self.registered_stages = registered_stages
        self.stage_instances = {}
        self._init_stage_instances()
    
    def _init_stage_instances(self):
        """Create temporary instances to analyze metadata."""
        for name, stage_class in self.registered_stages.items():
            try:
                # Initialize with empty config to avoid attribute errors
                self.stage_instances[name] = stage_class(stage_id=f"temp_{name}", config={})
            except Exception as e:
                logger.warning(f"Could not initialize {name}: {e}")
    
    def get_required_inputs(self, stage_name: str) -> List[str]:
        """Get required input names for a stage."""
        if stage_name not in self.stage_instances:
            return []
        
        stage = self.stage_instances[stage_name]
        return [inp.name for inp in stage.metadata.inputs if inp.required]
    
    def get_stage_outputs(self, stage_name: str) -> List[str]:
        """Get output names for a stage."""
        if stage_name not in self.stage_instances:
            return []
        
        stage = self.stage_instances[stage_name]
        return [out.name for out in stage.metadata.outputs]
    
    def can_satisfy_input(self, required_input: str, required_type: DataType, 
                         available_data: Dict[str, DataType]) -> bool:
        """Check if available data can satisfy a required input."""
        return required_input in available_data and available_data[required_input] == required_type
    
    def find_producers(self, data_type: DataType) -> List[str]:
        """Find all stages that can produce a specific data type."""
        producers = []
        for stage_name, stage in self.stage_instances.items():
            for output in stage.metadata.outputs:
                if output.data_type == data_type:
                    producers.append(stage_name)
                    break
        return producers
    
    def analyze_dependencies(self, target_stage: str, available_data: Dict[str, DataType]) -> Dict[str, Any]:
        """Analyze what's needed to run a target stage."""
        if target_stage not in self.stage_instances:
            return {"error": f"Unknown stage: {target_stage}"}
        
        stage = self.stage_instances[target_stage]
        analysis = {
            "stage": target_stage,
            "satisfied_inputs": [],
            "missing_inputs": [],
            "suggested_stages": [],
            "dependency_chain": []
        }
        
        for inp in stage.metadata.inputs:
            if inp.required:
                if self.can_satisfy_input(inp.name, inp.data_type, available_data):
                    analysis["satisfied_inputs"].append({
                        "name": inp.name,
                        "type": inp.data_type.value,
                        "description": inp.description
                    })
                else:
                    missing_input = {
                        "name": inp.name,
                        "type": inp.data_type.value,
                        "description": inp.description,
                        "producers": self.find_producers(inp.data_type)
                    }
                    analysis["missing_inputs"].append(missing_input)
                    analysis["suggested_stages"].extend(missing_input["producers"])
        
        # Remove duplicates from suggested stages
        analysis["suggested_stages"] = list(set(analysis["suggested_stages"]))
        
        return analysis


class EnhancedStagePlayground:
    """Enhanced interactive development environment for pipeline stages."""
    
    def __init__(self, data_dir: str = "data", cache_dir: str = "dev_cache"):
        """Initialize the enhanced playground.
        
        Args:
            data_dir: Directory containing sample data
            cache_dir: Directory for caching intermediate results
        """
        self.data_dir = Path(data_dir)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize config manager and register stages
        self.config_manager = ConfigManager()
        self._register_stages()
        
        # Initialize dependency resolver
        self.resolver = DependencyResolver(self.config_manager.registered_stages)
        
        # Cache for intermediate results and available data
        self.available_data = self._scan_available_data()
        
        logger.info(f"Enhanced StagePlayground initialized")
        logger.info(f"Data directory: {self.data_dir}")
        logger.info(f"Cache directory: {self.cache_dir}")
        logger.info(f"Device: {settings.device}")
        logger.info(f"Registered stages: {list(self.config_manager.registered_stages.keys())}")
    
    def _register_stages(self):
        """Register all available stages with the config manager."""
        stages = {
            "AudioPreprocessor": AudioPreprocessor,
            "VoiceSeparator": VoiceSeparator,
            "EnhancedVoiceSeparator": EnhancedVoiceSeparator,
            "SpeechEnhancer": SpeechEnhancer,
            "SpeakerDiarizer": SpeakerDiarizer,
            "SpeechRecognizer": SpeechRecognizer,
            "VoiceSynthesizer": VoiceSynthesizer,
            "AudioReconstructor": AudioReconstructor,
            "SepformerSeparator": SepformerSeparator
        }
        
        for name, stage_class in stages.items():
            try:
                self.config_manager.register_stage(stage_class, name)
                logger.debug(f"Registered stage: {name}")
            except Exception as e:
                logger.warning(f"Failed to register {name}: {e}")
    
    def _scan_available_data(self) -> Dict[str, DataType]:
        """Scan cache and data directories to determine what data is available."""
        available = {}
        
        # Check for audio files in data directory
        audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
        for ext in audio_extensions:
            audio_files = list(self.data_dir.glob(f"**/*{ext}"))
            if audio_files:
                available["file_path"] = DataType.FILE_PATH
                break
        
        # Check cached results
        for cache_dir in self.cache_dir.glob("*"):
            if cache_dir.is_dir():
                metadata_file = cache_dir / "metadata.json"
                if metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        stage_name = metadata.get("stage_class", metadata.get("stage", ""))
                        if stage_name in self.resolver.stage_instances:
                            # Add outputs from this stage to available data
                            stage_outputs = self.resolver.get_stage_outputs(stage_name)
                            stage_instance = self.resolver.stage_instances[stage_name]
                            for output in stage_instance.metadata.outputs:
                                available[output.name] = output.data_type
                    except Exception as e:
                        logger.debug(f"Could not read metadata from {cache_dir}: {e}")
        
        return available
    
    def list_stages(self) -> Dict[str, Dict[str, Any]]:
        """List available pipeline stages with their metadata."""
        stages_info = {}
        for name, stage_class in self.config_manager.registered_stages.items():
            try:
                instance = stage_class()
                metadata = instance.metadata
                stages_info[name] = {
                    "name": metadata.name,
                    "description": metadata.description,
                    "category": metadata.category,
                    "model_name": metadata.model_name,
                    "inputs": [{"name": inp.name, "type": inp.data_type.value, 
                              "required": inp.required, "description": inp.description} 
                             for inp in metadata.inputs],
                    "outputs": [{"name": out.name, "type": out.data_type.value, 
                               "description": out.description} 
                              for out in metadata.outputs]
                }
            except Exception as e:
                stages_info[name] = {"error": f"Failed to get metadata: {e}"}
        
        return stages_info
    
    def analyze_stage_dependencies(self, stage_name: str) -> Dict[str, Any]:
        """Analyze dependencies for a specific stage."""
        return self.resolver.analyze_dependencies(stage_name, self.available_data)
    
    def suggest_execution_path(self, target_stage: str) -> List[str]:
        """Suggest an execution path to satisfy dependencies for target stage."""
        analysis = self.analyze_stage_dependencies(target_stage)
        
        if not analysis.get("missing_inputs"):
            return []  # No dependencies needed
        
        # Simple heuristic: suggest stages that can produce missing inputs
        # In a more sophisticated version, this would do proper topological sorting
        suggested_path = []
        for missing_input in analysis["missing_inputs"]:
            producers = missing_input["producers"]
            if producers:
                # For now, suggest the first producer
                # TODO: More intelligent selection based on available data
                suggested_path.append(producers[0])
        
        return list(dict.fromkeys(suggested_path))  # Remove duplicates while preserving order
    
    def interactive_dependency_resolution(self, target_stage: str) -> bool:
        """Interactively guide user through dependency resolution."""
        analysis = self.analyze_stage_dependencies(target_stage)
        
        if not analysis.get("missing_inputs"):
            print(f"✅ All dependencies satisfied for {target_stage}")
            return True
        
        print(f"\n🔍 Analyzing dependencies for {target_stage}...")
        print(f"📋 Satisfied inputs: {len(analysis['satisfied_inputs'])}")
        print(f"❌ Missing inputs: {len(analysis['missing_inputs'])}")
        
        for missing in analysis["missing_inputs"]:
            print(f"\n❌ Missing: {missing['name']} ({missing['type']})")
            print(f"   Description: {missing['description']}")
            
            if missing["producers"]:
                print(f"   🏭 Can be produced by: {', '.join(missing['producers'])}")
                
                # Ask user which stage to run
                if len(missing["producers"]) == 1:
                    producer = missing["producers"][0]
                    choice = input(f"   ▶️  Run {producer} to produce this data? (y/n): ").lower()
                    if choice == 'y':
                        print(f"   🚀 Running {producer}...")
                        return self._run_suggested_stage(producer)
                else:
                    print(f"   🔢 Multiple options available:")
                    for i, producer in enumerate(missing["producers"], 1):
                        producer_analysis = self.analyze_stage_dependencies(producer)
                        deps_status = "✅ Ready" if not producer_analysis.get("missing_inputs") else f"❌ Needs {len(producer_analysis['missing_inputs'])} inputs"
                        print(f"      {i}. {producer} ({deps_status})")
                    
                    choice = input(f"   ▶️  Select stage to run (1-{len(missing['producers'])}, or 's' to skip): ")
                    if choice.isdigit() and 1 <= int(choice) <= len(missing["producers"]):
                        selected_producer = missing["producers"][int(choice) - 1]
                        print(f"   🚀 Running {selected_producer}...")
                        return self._run_suggested_stage(selected_producer)
                    elif choice.lower() == 's':
                        continue
            else:
                print(f"   ⚠️  No known stages can produce this data type")
        
        return False
    
    def _run_suggested_stage(self, stage_name: str) -> bool:
        """Run a suggested stage and update available data."""
        try:
            # Check if this stage also has dependencies
            analysis = self.analyze_stage_dependencies(stage_name)
            if analysis.get("missing_inputs"):
                print(f"   📋 {stage_name} also has dependencies...")
                if not self.interactive_dependency_resolution(stage_name):
                    return False
            
            # Run the stage
            result = asyncio.run(self.run_stage(stage_name, cache_result=True))
            
            # Update available data
            self.available_data = self._scan_available_data()
            print(f"   ✅ {stage_name} completed successfully")
            return True
            
        except Exception as e:
            print(f"   ❌ Failed to run {stage_name}: {e}")
            return False
    
    async def run_stage(self, stage_name: str, 
                       input_data: Optional[Dict[str, Any]] = None,
                       config: Optional[Dict[str, Any]] = None,
                       cache_result: bool = True) -> Any:
        """Run a pipeline stage using the new interface.
        
        Args:
            stage_name: Name of the stage to run
            input_data: Direct input data (if None, will try to auto-resolve)
            config: Configuration for the stage
            cache_result: Whether to cache the result
            
        Returns:
            Stage output
        """
        if stage_name not in self.config_manager.registered_stages:
            raise ValueError(f"Unknown stage: {stage_name}. Available: {list(self.config_manager.registered_stages.keys())}")
        
        stage_class = self.config_manager.registered_stages[stage_name]
        stage_instance = stage_class(stage_id=f"{stage_name}_playground", config=config or {})
        
        # Auto-resolve inputs if not provided
        if input_data is None:
            input_data = self._auto_resolve_inputs(stage_instance)
        
        logger.info(f"Running stage: {stage_name}")
        logger.info(f"Input keys: {list(input_data.keys()) if input_data else 'None'}")
        
        start_time = time.time()
        
        try:
            # Run the stage
            result = await stage_instance.process(input_data)
            
            end_time = time.time()
            duration = end_time - start_time
            
            logger.info(f"Stage {stage_name} completed in {duration:.2f}s")
            
            # Cache result if requested
            if cache_result:
                cache_key = f"{stage_name.lower()}_{int(start_time)}"
                self.save_result(cache_key, result, stage_name)
                logger.info(f"Result cached as: {cache_key}")
            
            return result
            
        except Exception as e:
            logger.error(f"Stage {stage_name} failed: {e}")
            raise
    
    def _auto_resolve_inputs(self, stage_instance: EnhancedPipelineStage) -> Dict[str, Any]:
        """Automatically resolve inputs for a stage from cached data."""
        inputs = {}
        
        for input_def in stage_instance.metadata.inputs:
            if input_def.required:
                # Try to find data that matches this input
                value = self._find_matching_data(input_def.name, input_def.data_type)
                if value is not None:
                    inputs[input_def.name] = value
                else:
                    raise ValueError(f"Could not resolve required input '{input_def.name}' of type {input_def.data_type.value}")
        
        return inputs
    
    def _find_matching_data(self, input_name: str, data_type: DataType) -> Any:
        """Find cached data that matches the required input."""
        # Special case for file_path - look for audio files
        if data_type == DataType.FILE_PATH:
            audio_extensions = ['.wav', '.mp3', '.flac', '.m4a']
            for ext in audio_extensions:
                audio_files = list(self.data_dir.glob(f"**/*{ext}"))
                if audio_files:
                    return str(audio_files[0])
        
        # Look through cached results for matching data
        for cache_dir in self.cache_dir.glob("*"):
            if cache_dir.is_dir():
                data_file = cache_dir / "data.pkl"
                metadata_file = cache_dir / "metadata.json"
                
                if data_file.exists() and metadata_file.exists():
                    try:
                        with open(metadata_file, 'r') as f:
                            metadata = json.load(f)
                        
                        stage_name = metadata.get("stage_class", metadata.get("stage", ""))
                        if stage_name in self.resolver.stage_instances:
                            stage_instance = self.resolver.stage_instances[stage_name]
                            
                            # Check if this stage produces the required output
                            for output in stage_instance.metadata.outputs:
                                if output.name == input_name and output.data_type == data_type:
                                    # Load and return the data
                                    with open(data_file, 'rb') as f:
                                        cached_data = pickle.load(f)
                                    
                                    # Extract the specific output if it's a dict
                                    if isinstance(cached_data, dict) and input_name in cached_data:
                                        return cached_data[input_name]
                                    elif output.name == "audio_with_sr" and isinstance(cached_data, tuple):
                                        return cached_data
                                    elif not isinstance(cached_data, dict):
                                        return cached_data
                    except Exception as e:
                        logger.debug(f"Could not load cached data from {cache_dir}: {e}")
        
        return None
    
    def save_result(self, name: str, data: Any, stage_name: str) -> Path:
        """Save result to cache with enhanced metadata."""
        # Create result directory
        result_dir = self.cache_dir / name
        result_dir.mkdir(exist_ok=True)
        
        # Save pickle data
        pickle_path = result_dir / "data.pkl"
        with open(pickle_path, 'wb') as f:
            pickle.dump(data, f)
        
        # Enhanced metadata
        metadata = {
            "timestamp": time.time(),
            "type": str(type(data)),
            "stage": stage_name.lower(),
            "stage_class": stage_name,
            "outputs": {}
        }
        
        # Add output type information
        if stage_name in self.resolver.stage_instances:
            stage_instance = self.resolver.stage_instances[stage_name]
            for output in stage_instance.metadata.outputs:
                if isinstance(data, dict) and output.name in data:
                    metadata["outputs"][output.name] = {
                        "type": output.data_type.value,
                        "description": output.description
                    }
                elif not isinstance(data, dict):
                    metadata["outputs"][output.name] = {
                        "type": output.data_type.value,
                        "description": output.description
                    }
        
        metadata_path = result_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save audio files if applicable
        self._save_audio_files(result_dir, data)
        
        logger.info(f"Saved result to: {result_dir}")
        return result_dir
    
    def _save_audio_files(self, result_dir: Path, data: Any) -> None:
        """Save audio data as files for easy inspection."""
        try:
            sample_rate = 44100  # Default sample rate
            
            if isinstance(data, tuple) and len(data) == 2:
                # Audio with sample rate: (audio_data, sample_rate)
                audio_data, sr = data
                if isinstance(audio_data, np.ndarray):
                    save_audio(audio_data, result_dir / "audio.wav", sr)
                    
            elif isinstance(data, dict):
                # Multiple audio outputs
                for key, value in data.items():
                    if isinstance(value, np.ndarray) and value.ndim >= 1:
                        filename = f"{key}.wav"
                        save_audio(value, result_dir / filename, sample_rate)
                    elif isinstance(value, (list, dict)) and key in ["transcription", "speaker_segments"]:
                        # Save text/metadata as JSON
                        json_path = result_dir / f"{key}.json"
                        with open(json_path, 'w') as f:
                            json.dump(value, f, indent=2, default=str)
                        
            elif isinstance(data, list):
                # Speaker segments or similar
                json_path = result_dir / "segments.json"
                with open(json_path, 'w') as f:
                    json.dump(data, f, indent=2, default=str)
                    
            elif isinstance(data, np.ndarray):
                # Raw audio array
                save_audio(data, result_dir / "audio.wav", sample_rate)
                
        except Exception as e:
            logger.warning(f"Could not save audio files: {e}")


async def main():
    """Enhanced CLI interface."""
    parser = argparse.ArgumentParser(description="Enhanced Pipeline Stage Development Playground")
    parser.add_argument("command", choices=["list", "run", "analyze", "deps", "auto"], 
                       help="Command to execute")
    parser.add_argument("--stage", "-s", help="Stage name to operate on")
    parser.add_argument("--config", "-c", help="JSON config string for stage")
    parser.add_argument("--data-dir", "-d", default="data", help="Data directory")
    parser.add_argument("--no-cache", action="store_true", help="Don't cache results")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive dependency resolution")
    
    args = parser.parse_args()
    
    playground = EnhancedStagePlayground(data_dir=args.data_dir)
    
    if args.command == "list":
        print("🔧 Available Pipeline Stages:")
        print("=" * 50)
        
        stages_info = playground.list_stages()
        for name, info in stages_info.items():
            if "error" in info:
                print(f"❌ {name}: {info['error']}")
                continue
                
            print(f"\n📦 {name}")
            print(f"   📝 {info['description']}")
            print(f"   🏷️  Category: {info['category']}")
            if info['model_name']:
                print(f"   🤖 Model: {info['model_name']}")
            
            print(f"   📥 Inputs ({len(info['inputs'])}):")
            for inp in info['inputs']:
                req_marker = "🔴" if inp['required'] else "🟡"
                print(f"      {req_marker} {inp['name']} ({inp['type']}): {inp['description']}")
            
            print(f"   📤 Outputs ({len(info['outputs'])}):")
            for out in info['outputs']:
                print(f"      🟢 {out['name']} ({out['type']}): {out['description']}")
    
    elif args.command == "analyze":
        if not args.stage:
            print("❌ Error: --stage required for analyze command")
            return
        
        print(f"🔍 Analyzing available data...")
        print(f"📊 Available data types: {list(playground.available_data.keys())}")
        
        analysis = playground.analyze_stage_dependencies(args.stage)
        if "error" in analysis:
            print(f"❌ {analysis['error']}")
            return
        
        print(f"\n🎯 Dependency Analysis for {args.stage}:")
        print("=" * 50)
        
        if analysis["satisfied_inputs"]:
            print(f"✅ Satisfied inputs ({len(analysis['satisfied_inputs'])}):")
            for inp in analysis["satisfied_inputs"]:
                print(f"   🟢 {inp['name']} ({inp['type']}): {inp['description']}")
        
        if analysis["missing_inputs"]:
            print(f"\n❌ Missing inputs ({len(analysis['missing_inputs'])}):")
            for inp in analysis["missing_inputs"]:
                print(f"   🔴 {inp['name']} ({inp['type']}): {inp['description']}")
                if inp["producers"]:
                    print(f"      🏭 Can be produced by: {', '.join(inp['producers'])}")
                else:
                    print(f"      ⚠️  No known producers")
        
        if analysis["suggested_stages"]:
            print(f"\n💡 Suggested stages to run first: {', '.join(analysis['suggested_stages'])}")
        else:
            print(f"\n🚀 Ready to run!")
    
    elif args.command == "deps":
        if not args.stage:
            print("❌ Error: --stage required for deps command")
            return
        
        suggested_path = playground.suggest_execution_path(args.stage)
        if suggested_path:
            print(f"📋 Suggested execution path for {args.stage}:")
            for i, stage in enumerate(suggested_path, 1):
                print(f"   {i}. {stage}")
        else:
            print(f"🚀 {args.stage} is ready to run!")
    
    elif args.command == "run":
        if not args.stage:
            print("❌ Error: --stage required for run command")
            return
        
        # Parse config if provided
        config = None
        if args.config:
            try:
                config = json.loads(args.config)
            except json.JSONDecodeError as e:
                print(f"❌ Invalid JSON config: {e}")
                return
        
        try:
            if args.interactive:
                # Interactive dependency resolution
                if playground.interactive_dependency_resolution(args.stage):
                    print(f"🚀 Now running {args.stage}...")
                    result = await playground.run_stage(
                        args.stage,
                        config=config,
                        cache_result=not args.no_cache
                    )
                    print(f"✅ {args.stage} completed successfully!")
                else:
                    print(f"❌ Could not resolve dependencies for {args.stage}")
            else:
                # Direct run
                result = await playground.run_stage(
                    args.stage,
                    config=config,
                    cache_result=not args.no_cache
                )
                print(f"✅ {args.stage} completed successfully!")
            
        except Exception as e:
            print(f"❌ Error running {args.stage}: {e}")
    
    elif args.command == "auto":
        if not args.stage:
            print("❌ Error: --stage required for auto command")
            return
        
        print(f"🤖 Auto-resolving dependencies for {args.stage}...")
        
        # Auto-resolve and run suggested stages
        suggested_path = playground.suggest_execution_path(args.stage)
        if suggested_path:
            print(f"📋 Will run: {' → '.join(suggested_path + [args.stage])}")
            
            for stage in suggested_path:
                print(f"\n🚀 Running {stage}...")
                try:
                    await playground.run_stage(stage, cache_result=True)
                    print(f"✅ {stage} completed")
                except Exception as e:
                    print(f"❌ {stage} failed: {e}")
                    return
        
        # Finally run the target stage
        print(f"\n🎯 Running target stage: {args.stage}...")
        try:
            result = await playground.run_stage(args.stage, cache_result=not args.no_cache)
            print(f"🎉 All stages completed successfully!")
        except Exception as e:
            print(f"❌ Target stage failed: {e}")


if __name__ == "__main__":
    asyncio.run(main())
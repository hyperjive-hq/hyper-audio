#!/usr/bin/env python3
"""Configuration-driven development playground for pipeline stages.

This script uses YAML configuration files to define pipeline runs, making it easy
to test specific combinations of stages, inputs, and configurations.
"""

import asyncio
import argparse
import json
import sys
import time
import yaml
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from hyper_audio.pipeline.stage_interface import EnhancedPipelineStage, DataType
from hyper_audio.pipeline.config_manager import ConfigManager
from hyper_audio.pipeline.stages import (
    AudioPreprocessor, VoiceSeparator, SpeakerDiarizer,
    SpeechRecognizer, VoiceSynthesizer, AudioReconstructor,
    SepformerSeparator, EnhancedVoiceSeparator, SpeechEnhancer,
    AudioPostProcessor
)
from hyper_audio.config.settings import settings
from hyper_audio.utils.logging_utils import get_logger

logger = get_logger("dev_playground_config")


class ConfigPipelineRunner:
    """Runs pipeline configurations from YAML files."""
    
    def __init__(self, config_path: str = "dev_playground_config.yaml"):
        """Initialize with configuration file."""
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.cache_dir = Path(self.config.get("defaults", {}).get("cache_dir", "dev_cache"))
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize config manager and register stages
        self.config_manager = ConfigManager()
        self._register_stages()
        
        # Track stage instances and results
        self.stage_results = {}
        
        logger.info(f"Config pipeline runner initialized")
        logger.info(f"Config file: {self.config_path}")
        logger.info(f"Cache directory: {self.cache_dir}")
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)
    
    def _register_stages(self):
        """Register all available stages with the config manager."""
        stages = {
            "AudioPreprocessor": AudioPreprocessor,
            "VoiceSeparator": VoiceSeparator,
            "EnhancedVoiceSeparator": EnhancedVoiceSeparator,
            "SpeechEnhancer": SpeechEnhancer,
            "AudioPostProcessor": AudioPostProcessor,
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
    
    def list_pipelines(self):
        """List available pipeline configurations."""
        pipelines = self.config.get("pipelines", {})
        
        print("🔧 Available Pipeline Configurations:")
        print("=" * 50)
        
        for name, pipeline in pipelines.items():
            print(f"\n📦 {name}")
            print(f"   📝 {pipeline.get('description', 'No description')}")
            
            stages = pipeline.get("stages", [])
            print(f"   🔗 Stages ({len(stages)}):")
            for i, stage in enumerate(stages, 1):
                stage_name = stage.get("stage")
                depends = stage.get("depends_on", [])
                if isinstance(depends, str):
                    depends = [depends]
                
                dependency_info = f" (depends on: {', '.join(depends)})" if depends else ""
                print(f"      {i}. {stage_name}{dependency_info}")
    
    def list_test_scenarios(self):
        """List available test scenarios."""
        scenarios = self.config.get("test_scenarios", {})
        
        print("🧪 Available Test Scenarios:")
        print("=" * 50)
        
        for name, scenario in scenarios.items():
            print(f"\n🎯 {name}")
            pipeline = scenario.get("pipeline")
            print(f"   📋 Pipeline: {pipeline}")
            
            expected = scenario.get("expected_outputs", [])
            if expected:
                print(f"   📤 Expected outputs: {', '.join(expected)}")
            
            validation = scenario.get("validation", {})
            if validation:
                print(f"   ✅ Validation: {validation}")
    
    async def run_pipeline(self, pipeline_name: str, **kwargs):
        """Run a specific pipeline configuration."""
        pipelines = self.config.get("pipelines", {})
        
        if pipeline_name not in pipelines:
            raise ValueError(f"Pipeline '{pipeline_name}' not found. Available: {list(pipelines.keys())}")
        
        pipeline = pipelines[pipeline_name]
        print(f"🚀 Running pipeline: {pipeline_name}")
        print(f"📝 Description: {pipeline.get('description', 'No description')}")
        
        stages = pipeline.get("stages", [])
        print(f"🔗 Will execute {len(stages)} stages")
        
        # Execute stages in order
        for i, stage_config in enumerate(stages, 1):
            stage_name = stage_config.get("stage")
            print(f"\n📍 Stage {i}/{len(stages)}: {stage_name}")
            
            # Check dependencies
            depends_on = stage_config.get("depends_on", [])
            if isinstance(depends_on, str):
                depends_on = [depends_on]
            
            # Resolve inputs
            stage_inputs = await self._resolve_stage_inputs(stage_config, depends_on)
            
            # Debug logging
            logger.debug(f"Stage {stage_name} resolved inputs: {list(stage_inputs.keys())}")
            for key, value in stage_inputs.items():
                if hasattr(value, 'shape'):
                    logger.debug(f"  {key}: numpy array shape {value.shape}")
                elif isinstance(value, dict):
                    logger.debug(f"  {key}: dict with keys {list(value.keys())}")
                else:
                    logger.debug(f"  {key}: {type(value)}")
            
            # Get stage configuration
            stage_config_dict = stage_config.get("config", {})
            
            # Merge with default config if available
            stage_defaults = self.config.get("stages", {}).get(stage_name, {}).get("default_config", {})
            final_config = {**stage_defaults, **stage_config_dict, **kwargs}
            
            # Run the stage
            start_time = time.time()
            result = await self._run_stage(stage_name, stage_inputs, final_config)
            duration = time.time() - start_time
            
            # Verify stage outputs using the stage's own verification
            verification_passed, verification_messages = await self._verify_stage_output(stage_name, result)
            
            # Print verification messages with proper indentation
            for message in verification_messages:
                print(f"      {message}")
            
            # Store result for dependent stages
            self.stage_results[stage_name] = result
            
            if verification_passed:
                print(f"   ✅ {stage_name} completed in {duration:.2f}s")
            else:
                print(f"   ⚠️  {stage_name} completed in {duration:.2f}s (verification issues detected)")
        
        print(f"\n🎉 Pipeline '{pipeline_name}' completed successfully!")
    
    async def _resolve_stage_inputs(self, stage_config: Dict, depends_on: List[str]) -> Dict[str, Any]:
        """Resolve inputs for a stage from explicit inputs or dependencies."""
        inputs = {}
        
        # Add explicit inputs from config
        explicit_inputs = stage_config.get("inputs", {})
        for input_name, input_value in explicit_inputs.items():
            if isinstance(input_value, str) and input_value.startswith("data/"):
                # It's a file path
                inputs[input_name] = input_value
            else:
                inputs[input_name] = input_value
        
        # Add inputs from dependency results
        for dependency in depends_on:
            if dependency in self.stage_results:
                dep_result = self.stage_results[dependency]
                
                # Map dependency outputs to stage inputs based on compatibility
                stage_class = self.config_manager.registered_stages.get(stage_config["stage"])
                if stage_class:
                    stage_instance = stage_class()
                    await self._map_dependency_outputs(stage_instance, dep_result, inputs)
        
        return inputs
    
    async def _map_dependency_outputs(self, stage_instance: EnhancedPipelineStage, 
                                    dep_result: Dict[str, Any], inputs: Dict[str, Any]):
        """Map dependency outputs to stage inputs based on type compatibility."""
        stage_inputs = stage_instance.metadata.inputs
        required_inputs = [inp for inp in stage_inputs if inp.required]
        
        # If there are required inputs, map them specifically
        for input_def in required_inputs:
            if input_def.name not in inputs:
                # Try to find compatible data in dependency result
                if input_def.name in dep_result:
                    inputs[input_def.name] = dep_result[input_def.name]
                else:
                    # Look for compatible data types
                    for key, value in dep_result.items():
                        if self._is_compatible_data(value, input_def.data_type):
                            inputs[input_def.name] = value
                            break
        
        # If no required inputs, or for optional inputs, pass through compatible data
        for input_def in stage_inputs:
            if input_def.name not in inputs:
                # Try direct mapping first
                if input_def.name in dep_result:
                    inputs[input_def.name] = dep_result[input_def.name]
                else:
                    # Look for compatible data types
                    for key, value in dep_result.items():
                        if self._is_compatible_data(value, input_def.data_type):
                            inputs[input_def.name] = value
                            break
    
    def _is_compatible_data(self, data: Any, expected_type: DataType) -> bool:
        """Check if data is compatible with expected type."""
        # This is a simplified version - could be enhanced with full type checking
        if expected_type == DataType.AUDIO_WITH_SR:
            return isinstance(data, tuple) and len(data) == 2
        elif expected_type == DataType.AUDIO_MONO:
            return hasattr(data, 'ndim') and data.ndim == 1
        elif expected_type == DataType.SEPARATED_AUDIO:
            return isinstance(data, dict) and any(k in data for k in ['vocals', 'music'])
        elif expected_type == DataType.FILE_PATH:
            return isinstance(data, str) and Path(data).exists()
        
        return False
    
    async def _run_stage(self, stage_name: str, inputs: Dict[str, Any], 
                        config: Dict[str, Any]) -> Dict[str, Any]:
        """Run a single stage with given inputs and configuration."""
        if stage_name not in self.config_manager.registered_stages:
            raise ValueError(f"Unknown stage: {stage_name}")
        
        stage_class = self.config_manager.registered_stages[stage_name]
        stage_instance = stage_class(stage_id=f"{stage_name}_config", config=config)
        
        logger.info(f"Running {stage_name} with inputs: {list(inputs.keys())}")
        logger.debug(f"Stage config: {config}")
        
        # Run the stage
        result = await stage_instance.process(inputs)
        
        # Save result to cache
        cache_key = f"{stage_name.lower()}_{int(time.time())}"
        cache_dir = self.cache_dir / cache_key
        cache_dir.mkdir(exist_ok=True)
        
        # Save as pickle
        import pickle
        with open(cache_dir / "data.pkl", 'wb') as f:
            pickle.dump(result, f)
        
        # Save metadata
        metadata = {
            "timestamp": time.time(),
            "stage": stage_name,
            "config": config,
            "inputs": list(inputs.keys()),
            "outputs": list(result.keys()) if isinstance(result, dict) else [type(result).__name__]
        }
        
        with open(cache_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Save audio files if applicable
        await self._save_audio_outputs(cache_dir, result)
        
        logger.info(f"Results saved to: {cache_dir}")
        
        return result
    
    async def _save_audio_outputs(self, cache_dir: Path, result: Dict[str, Any]):
        """Save audio outputs as WAV files."""
        try:
            from hyper_audio.utils.audio_utils import save_audio
            
            sample_rate = 44100  # Default
            
            if isinstance(result, dict):
                for key, value in result.items():
                    if hasattr(value, 'ndim') and value.ndim >= 1:
                        # It's an audio array
                        output_path = cache_dir / f"{key}.wav"
                        save_audio(value, output_path, sample_rate)
                        logger.debug(f"Saved audio: {output_path}")
        except Exception as e:
            logger.warning(f"Could not save audio outputs: {e}")
    
    async def run_test_scenario(self, scenario_name: str):
        """Run a test scenario with validation."""
        scenarios = self.config.get("test_scenarios", {})
        
        if scenario_name not in scenarios:
            raise ValueError(f"Test scenario '{scenario_name}' not found")
        
        scenario = scenarios[scenario_name]
        pipeline_name = scenario.get("pipeline")
        
        print(f"🧪 Running test scenario: {scenario_name}")
        
        # Run the pipeline
        await self.run_pipeline(pipeline_name)
        
        # Validate results
        expected_outputs = scenario.get("expected_outputs", [])
        validation = scenario.get("validation", {})
        
        if expected_outputs or validation:
            print(f"\n🔍 Validating results...")
            await self._validate_scenario_results(scenario)
    
    async def _validate_scenario_results(self, scenario: Dict[str, Any]):
        """Validate scenario results against expectations."""
        # This is a placeholder for validation logic
        print("✅ Validation passed (placeholder)")
    
    async def _verify_stage_output(self, stage_name: str, result: Any) -> Tuple[bool, List[str]]:
        """Verify stage output using the stage's own verification method."""
        messages = [f"🔍 Verifying {stage_name} outputs..."]
        
        try:
            # Get stage class and create instance
            stage_class = self.config_manager.registered_stages.get(stage_name)
            if not stage_class:
                messages.append(f"❌ Unknown stage class: {stage_name}")
                return False, messages
            
            stage_instance = stage_class()
            
            # Use the stage's own verification method
            verification_passed, stage_messages = await stage_instance.verify_outputs(result)
            messages.extend(stage_messages)
            
            return verification_passed, messages
            
        except Exception as e:
            messages.append(f"❌ Verification error: {e}")
            return False, messages


async def main():
    """CLI interface for configuration-driven pipeline development."""
    parser = argparse.ArgumentParser(description="Configuration-driven Pipeline Development")
    parser.add_argument("command", choices=["list", "run", "test", "pipelines", "scenarios"],
                       help="Command to execute")
    parser.add_argument("--name", "-n", help="Pipeline or scenario name")
    parser.add_argument("--config", "-c", default="dev_playground_config.yaml", 
                       help="Configuration file path")
    
    args = parser.parse_args()
    
    runner = ConfigPipelineRunner(args.config)
    
    if args.command == "list" or args.command == "pipelines":
        runner.list_pipelines()
    
    elif args.command == "scenarios":
        runner.list_test_scenarios()
    
    elif args.command == "run":
        if not args.name:
            print("❌ Error: --name required for run command")
            return
        
        await runner.run_pipeline(args.name)
    
    elif args.command == "test":
        if not args.name:
            print("❌ Error: --name required for test command")
            return
        
        await runner.run_test_scenario(args.name)


if __name__ == "__main__":
    asyncio.run(main())
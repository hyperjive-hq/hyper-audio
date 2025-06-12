#!/usr/bin/env python3
"""Example of using the enhanced configurable pipeline system."""

import asyncio
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hyper_audio.pipeline.config_manager import ConfigManager
from hyper_audio.pipeline.stages.enhanced_separator import EnhancedVoiceSeparator, SpeechEnhancer
from hyper_audio.pipeline.stages.preprocessor import AudioPreprocessor


async def main():
    """Demonstrate the enhanced pipeline system."""
    
    # Initialize configuration manager
    config_manager = ConfigManager()
    
    # Register available stages
    config_manager.register_stage(AudioPreprocessor, "AudioPreprocessor")
    config_manager.register_stage(EnhancedVoiceSeparator, "VoiceSeparator")
    config_manager.register_stage(SpeechEnhancer, "SpeechEnhancer")
    
    print("🔧 Registered Stages:")
    for name in config_manager.registered_stages:
        print(f"  - {name}")
    
    # Load configuration
    config_path = Path(__file__).parent / "multi_stage_config.yaml"
    try:
        pipeline_config = config_manager.load_config(config_path)
        print(f"\n📋 Loaded Configuration: {pipeline_config.name}")
        print(f"Description: {pipeline_config.description}")
        
        # Show stages
        print(f"\n🏭 Configured Stages:")
        for stage in pipeline_config.stages:
            if stage.enabled:
                print(f"  ✅ {stage.stage_id} ({stage.stage_class})")
            else:
                print(f"  ❌ {stage.stage_id} ({stage.stage_class}) - DISABLED")
        
        # Show connections
        print(f"\n🔗 Connections:")
        for conn in pipeline_config.connections:
            print(f"  {conn.from_stage}.{conn.from_output} → {conn.to_stage}.{conn.to_input}")
        
        # Validate configuration
        print(f"\n✅ Validating Configuration...")
        is_valid, errors = config_manager.validate_config(pipeline_config)
        
        if is_valid:
            print("✅ Configuration is valid!")
            
            # Show execution order
            execution_order = config_manager.get_execution_order(pipeline_config)
            print(f"\n🚀 Execution Order: {' → '.join(execution_order)}")
            
        else:
            print("❌ Configuration has errors:")
            for error in errors:
                print(f"  • {error}")
                
    except FileNotFoundError:
        print(f"❌ Config file not found: {config_path}")
        print("Creating example configuration...")
        
        # Create and save example config
        example_config = config_manager.create_example_config()
        config_manager.save_config(example_config, config_path)
        print(f"✅ Example config saved to: {config_path}")
    
    # Demonstrate stage introspection
    print(f"\n🔍 Stage Capabilities:")
    
    # Create example stage instances
    voice_separator = EnhancedVoiceSeparator("test_separator")
    speech_enhancer = SpeechEnhancer("test_enhancer")
    
    print(f"\n📊 {voice_separator._metadata.name}:")
    print(f"  Category: {voice_separator._metadata.category}")
    print(f"  Inputs: {[inp.name + f' ({inp.data_type.value})' for inp in voice_separator._metadata.inputs]}")
    print(f"  Outputs: {[out.name + f' ({out.data_type.value})' for out in voice_separator._metadata.outputs]}")
    
    print(f"\n📊 {speech_enhancer._metadata.name}:")
    print(f"  Category: {speech_enhancer._metadata.category}")
    print(f"  Inputs: {[inp.name + f' ({inp.data_type.value})' for inp in speech_enhancer._metadata.inputs]}")
    print(f"  Outputs: {[out.name + f' ({out.data_type.value})' for out in speech_enhancer._metadata.outputs]}")
    
    # Test stage compatibility
    print(f"\n🔗 Stage Compatibility:")
    connections = voice_separator.can_connect_to(speech_enhancer)
    if connections:
        print(f"  {voice_separator._metadata.name} can connect to {speech_enhancer._metadata.name}:")
        for output, compatible_inputs in connections.items():
            print(f"    {output} → {compatible_inputs}")
    else:
        print(f"  {voice_separator._metadata.name} cannot connect to {speech_enhancer._metadata.name}")


if __name__ == "__main__":
    asyncio.run(main())
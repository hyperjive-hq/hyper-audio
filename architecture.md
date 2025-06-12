# Hyper-Audio Pipeline Architecture

## Overview

The Hyper-Audio project implements a **resilient and configurable audio processing pipeline** designed for voice replacement in audio files. The system features a modular, stage-based architecture with comprehensive checkpointing, failure recovery, performance monitoring, and a powerful configuration management system that supports complex multi-stage audio processing workflows.

## High-Level Architecture

```mermaid
graph TB
    A[Input Audio File] --> B[ConfigManager]
    B --> C[Pipeline Configuration]
    C --> D[ResilientAudioPipeline]
    D --> E[Stage Execution Engine]
    E --> F[Output Audio File]
    
    B --> G[Dependency Validation]
    B --> H[Type Safety Checking]
    
    E --> I[Checkpoint Manager]
    E --> J[State Manager]
    E --> K[Analytics Engine]
    
    I --> L[(Checkpoint Storage)]
    J --> M[(Pipeline State)]
    K --> N[(Performance Metrics)]
    
    O[Error Handler] --> P[Failure Reports]
    E --> O
    
    Q[Stage Registry] --> B
    R[YAML/JSON Config] --> B
```

## Enhanced Pipeline Architecture

### Configurable Pipeline Flow

The enhanced pipeline supports flexible, configurable workflows beyond the traditional linear sequence. It allows for complex processing chains with multiple branches and specialized stages:

```mermaid
flowchart TD
    A[Input Audio] --> B[AudioPreprocessor]
    B --> C[VoiceSeparator]
    C --> D[SpeechEnhancer]
    C --> E[SpeakerDiarizer]
    D --> F[SpeechRecognizer]
    E --> F
    F --> G[VoiceSynthesizer]
    G --> H[AudioReconstructor]
    H --> I[Output Audio]
    
    B -.-> CP1[(Checkpoint)]
    C -.-> CP2[(Checkpoint)]
    D -.-> CP3[(Checkpoint)]
    E -.-> CP4[(Checkpoint)]
    F -.-> CP5[(Checkpoint)]
    G -.-> CP6[(Checkpoint)]
    H -.-> CP7[(Checkpoint)]
```

### Traditional Linear Pipeline (Original)

For comparison, the original linear pipeline flow:

```mermaid
flowchart LR
    A[Input Audio] --> B[AudioPreprocessor]
    B --> C[VoiceSeparator]
    C --> D[SpeakerDiarizer]
    D --> E[SpeechRecognizer]
    E --> F[VoiceSynthesizer]
    F --> G[AudioReconstructor]
    G --> H[Output Audio]
    
    B -.-> CP1[(Checkpoint)]
    C -.-> CP2[(Checkpoint)]
    D -.-> CP3[(Checkpoint)]
    E -.-> CP4[(Checkpoint)]
    F -.-> CP5[(Checkpoint)]
    G -.-> CP6[(Checkpoint)]
```

## Enhanced Stage Interface System

### Type-Safe Stage Architecture

The enhanced pipeline introduces a sophisticated stage interface system with compile-time type safety and automatic compatibility checking:

```mermaid
classDiagram
    class EnhancedPipelineStage {
        <<abstract>>
        +stage_id: str
        +config: Dict
        +_metadata: StageMetadata
        +get_metadata() StageMetadata
        +process(inputs) Dict
        +validate_inputs(inputs) bool
        +can_connect_to(other_stage) Dict
    }
    
    class StageMetadata {
        +name: str
        +description: str
        +category: str
        +model_name: str
        +inputs: List[StageInput]
        +outputs: List[StageOutput]
        +performance_notes: str
    }
    
    class StageInput {
        +name: str
        +data_type: DataType
        +required: bool
        +description: str
    }
    
    class StageOutput {
        +name: str
        +data_type: DataType
        +description: str
    }
    
    class DataType {
        <<enumeration>>
        AUDIO_MONO
        AUDIO_STEREO
        AUDIO_WITH_SR
        SEPARATED_AUDIO
        SPEAKER_SEGMENTS
        TRANSCRIPTION
        SYNTHESIZED_AUDIO
    }
    
    EnhancedPipelineStage --> StageMetadata
    StageMetadata --> StageInput
    StageMetadata --> StageOutput
    StageInput --> DataType
    StageOutput --> DataType
```

### Configuration Management System

```mermaid
classDiagram
    class ConfigManager {
        +registered_stages: Dict
        +register_stage(stage_class, name)
        +load_config(path) PipelineConfig
        +save_config(config, path)
        +validate_config(config) Tuple[bool, List[str]]
        +get_execution_order(config) List[str]
        +create_example_config() PipelineConfig
    }
    
    class PipelineConfig {
        +name: str
        +description: str
        +global_config: Dict
        +stages: List[StageConfig]
        +connections: List[StageConnection]
    }
    
    class StageConfig {
        +stage_class: str
        +stage_id: str
        +enabled: bool
        +config: Dict
    }
    
    class StageConnection {
        +from_stage: str
        +from_output: str
        +to_stage: str
        +to_input: str
    }
    
    ConfigManager --> PipelineConfig
    PipelineConfig --> StageConfig
    PipelineConfig --> StageConnection
```

## Stage Specifications

### 1. Audio Preprocessor
**Purpose**: Audio preprocessing and normalization
- **Input**: Raw audio file path
- **Output**: Normalized audio data + sample rate
- **Functionality**: 
  - Audio format standardization
  - Noise reduction
  - Volume normalization
  - Sample rate conversion

### 2. Enhanced Voice Separator
**Purpose**: Voice/music separation using advanced source separation models
- **Input**: Normalized audio + sample rate (DataType.AUDIO_WITH_SR)
- **Output**: Multiple separation outputs
  - `vocals`: Separated vocal audio (DataType.AUDIO_MONO)
  - `music`: Separated music/instrumental audio (DataType.AUDIO_MONO)
  - `separated_audio`: Dictionary with vocals and music keys (DataType.SEPARATED_AUDIO)
- **Model**: HTDemucs Fine-tuned (`htdemucs_ft`) - improved quality over standard HTDemucs
- **Functionality**:
  - High-quality vocal isolation from background music
  - Support for both mono and stereo audio
  - Automatic format conversion and validation
  - GPU-accelerated processing with CUDA support

### 3. Speech Enhancer (New)
**Purpose**: Noise reduction and speech quality enhancement
- **Input**: Audio with sample rate or separated audio (DataType.AUDIO_WITH_SR)
- **Output**: 
  - `enhanced_audio`: Noise-reduced audio (DataType.AUDIO_MONO)
  - `noise_estimate`: Estimated noise component (DataType.AUDIO_MONO)
- **Model**: SpeechBrain SepFormer (`speechbrain/sepformer-whamr`)
- **Functionality**:
  - Advanced noise reduction for speech clarity
  - Handles mixed content (music + noise + speech)
  - Configurable enhancement levels (moderate, aggressive)
  - Works as post-processing after music separation

### 4. Speaker Diarizer
**Purpose**: Speaker identification and segmentation
- **Input**: Enhanced vocal tracks + sample rate (DataType.AUDIO_WITH_SR)
- **Output**: Speaker segments with timestamps and IDs (DataType.SPEAKER_SEGMENTS)
- **Model**: Pyannote speaker-diarization (implemented in `transcribe.py`)
- **Functionality**:
  - Identify different speakers in audio
  - Generate time-based speaker segments
  - Assign unique speaker IDs
  - Configurable min/max speaker counts

### 5. Speech Recognizer
**Purpose**: Speech-to-text transcription with speaker attribution
- **Input**: 
  - Enhanced vocal tracks (DataType.AUDIO_WITH_SR)
  - Speaker segments (DataType.SPEAKER_SEGMENTS)
- **Output**: Transcription with speaker-specific text (DataType.TRANSCRIPTION)
- **Model**: Whisper large-v2 (implemented in `transcribe.py`)
- **Functionality**:
  - Convert speech to text with precise timing
  - Maintain speaker attribution from diarization
  - Support automatic language detection
  - High-accuracy transcription with confidence scores

### 6. Voice Synthesizer
**Purpose**: Voice synthesis and replacement
- **Input**: 
  - Transcription data (DataType.TRANSCRIPTION)
  - Target speaker configuration
  - Replacement voice reference
- **Output**: Synthesized audio for target speaker (DataType.SYNTHESIZED_AUDIO)
- **Model**: MetaVoice-1B (implemented in `tts.py`)
- **Functionality**:
  - Generate speech from text with voice cloning
  - Apply voice style transfer to match reference
  - Preserve natural prosody and timing
  - Support for multiple voice targets

### 7. Audio Reconstructor
**Purpose**: Final audio reconstruction and mixing
- **Input**: 
  - Original separated audio tracks
  - Synthesized replacement audio
  - Speaker segments and timing data
- **Output**: Final reconstructed audio with voice replacement
- **Functionality**:
  - Seamlessly replace target speaker segments
  - Mix synthesized speech with background music
  - Maintain precise timing and synchronization
  - Apply audio mastering and quality enhancement

## Data Models

### Pipeline State Model
```mermaid
classDiagram
    class PipelineState {
        +String job_id
        +String input_path
        +String output_path
        +Dict config
        +int current_stage
        +List stages_completed
        +List stage_metrics
        +DateTime created_at
        +DateTime updated_at
        +markStageCompleted()
        +addStageMetrics()
        +getOverallStatus()
    }
    
    class StageMetrics {
        +String stage_name
        +StageStatus status
        +DateTime start_time
        +DateTime end_time
        +float duration_seconds
        +float memory_peak_gb
        +String error_message
        +int retry_count
        +float checkpoint_size_mb
    }
    
    class PipelineResult {
        +String job_id
        +CheckpointManager checkpoint_manager
        +Any original_audio
        +int sample_rate
        +Dict separated_audio
        +List speaker_segments
        +Dict transcription
        +Dict synthesized_audio
        +Any final_audio
        +getStageData()
        +saveStageData()
    }
    
    PipelineState --> StageMetrics
    PipelineResult --> CheckpointManager
```

## Infrastructure Components

### Checkpoint Management
```mermaid
graph LR
    A[Stage Output] --> B[CheckpointManager]
    B --> C[Pickle Serialization]
    C --> D[Checksum Calculation]
    D --> E[(File Storage)]
    
    F[Stage Recovery] --> G[CheckpointManager]
    G --> H[Integrity Verification]
    H --> I[Data Deserialization]
    I --> J[Restored Stage Data]
```

**Features**:
- Automatic data serialization using pickle
- SHA256 checksum verification for data integrity
- Lazy loading of checkpoint data
- Automatic cleanup capabilities

### State Management
```mermaid
graph TB
    A[Pipeline Execution] --> B[StateManager]
    B --> C[JSON Serialization]
    C --> D[(State File)]
    
    E[Pipeline Resume] --> F[StateManager]
    F --> G[State Loading]
    G --> H[Pipeline Continuation]
```

**Features**:
- JSON-based state persistence
- Resume capability from any stage
- Metadata tracking (timestamps, progress, errors)
- Configuration preservation

### Retry and Error Handling
```mermaid
flowchart TD
    A[Stage Execution] --> B{Success?}
    B -->|Yes| C[Save Checkpoint]
    B -->|No| D[Increment Retry Count]
    D --> E{Max Retries?}
    E -->|No| F[Wait Delay]
    F --> G[Clear GPU Memory]
    G --> A
    E -->|Yes| H[Generate Failure Report]
    H --> I[Pipeline Failure]
    
    C --> J[Continue to Next Stage]
```

**Configuration**:
- Default max retries: 3
- Default retry delay: 5 seconds
- GPU memory cleanup between retries
- Detailed failure reporting

### Analytics and Monitoring
```mermaid
graph TB
    A[Pipeline Execution] --> B[Metrics Collection]
    B --> C[Performance Data]
    B --> D[Resource Usage]
    B --> E[Error Patterns]
    
    C --> F[PipelineAnalytics]
    D --> F
    E --> F
    
    F --> G[Job Reports]
    F --> H[Aggregate Analytics]
    F --> I[Performance Dashboards]
    F --> J[Recommendations]
```

**Capabilities**:
- Real-time job status tracking
- Performance metrics (duration, memory usage)
- Failure pattern analysis
- Automated recommendations
- Visual performance dashboards

## Key Features

### Resilience
- **Checkpointing**: Automatic stage-by-stage data persistence
- **Resume Capability**: Continue from any failed stage
- **Retry Logic**: Configurable retry attempts with exponential backoff
- **Error Recovery**: Detailed failure analysis and reporting

### Performance
- **GPU Memory Management**: Automatic cleanup and optimization
- **Lazy Loading**: Checkpoint data loaded only when needed
- **Resource Monitoring**: Real-time memory and performance tracking
- **Bottleneck Identification**: Automated performance analysis

### Scalability
- **Modular Design**: Independent, replaceable stages
- **Configuration Management**: Flexible pipeline configuration
- **Job Management**: Multiple concurrent pipeline jobs
- **Analytics**: Comprehensive performance monitoring

## Configuration System

### Multi-Stage Pipeline Configuration

The enhanced pipeline supports complex configurations through YAML/JSON files. Example multi-stage speech enhancement configuration:

```yaml
name: "Multi-Stage Speech Enhancement"
description: "Combines music separation and speech enhancement for optimal results"

global_config:
  device: "cuda"
  cache_dir: "~/.cache/hyper_audio"
  max_workers: 4

stages:
  - stage_class: "AudioPreprocessor"
    stage_id: "preprocessor"
    enabled: true
    config:
      target_sample_rate: 44100
      normalize: true
      
  - stage_class: "VoiceSeparator"
    stage_id: "music_separator"
    enabled: true
    config:
      model_name: "htdemucs_ft"
      overlap: 0.75
      
  - stage_class: "SpeechEnhancer"
    stage_id: "noise_reducer"
    enabled: true
    config:
      model_name: "speechbrain/sepformer-whamr"
      enhancement_level: "aggressive"

connections:
  - from_stage: "preprocessor"
    from_output: "audio_with_sr"
    to_stage: "music_separator"
    to_input: "audio_input"
    
  - from_stage: "music_separator"
    from_output: "vocals"
    to_stage: "noise_reducer"
    to_input: "audio_input"
```

### Key Configuration Features

1. **Dependency Validation**: Automatic checking of stage input/output compatibility
2. **Execution Ordering**: Topological sort for optimal stage execution sequence
3. **Type Safety**: Compile-time validation of data type connections
4. **Flexible Branching**: Support for parallel processing paths
5. **Dynamic Stage Registry**: Runtime registration of available stages

## Current Implementation Status

### ✅ Implemented Components
- **Pipeline Infrastructure**: Complete core pipeline with retry logic
- **Checkpoint System**: Full checkpointing and state management
- **Analytics Engine**: Comprehensive monitoring and reporting
- **Enhanced Configuration**: Type-safe pipeline configuration with dependency validation
- **Stage Interface System**: Strongly-typed stage inputs/outputs with compatibility checking
- **Multi-Stage Processing**: Support for complex processing chains (music separation + speech enhancement)
- **Development Playground**: Complete development environment for individual stage testing

### ✅ Enhanced Stage Implementations
- **Enhanced Voice Separator**: HTDemucs integration with type safety and GPU optimization
- **Speech Enhancer**: SpeechBrain SepFormer integration for noise reduction
- **Configuration Manager**: Complete system for pipeline configuration and validation
- **Model Loading Utilities**: Reusable model loading with progress feedback

### ⚠️ Partially Implemented
- **Stage Integration**: Some stage interfaces defined but full implementations pending
- **Standalone Scripts**: Individual processing scripts exist but need enhanced pipeline integration
  - `transcribe.py`: Diarization + Recognition functionality
  - `tts.py`: Voice synthesis using MetaVoice-1B

### 🔄 Integration Opportunities
- **Speaker Diarizer**: Integrate `transcribe.py` diarization → Enhanced SpeakerDiarizer
- **Speech Recognizer**: Integrate `transcribe.py` transcription → Enhanced SpeechRecognizer
- **Voice Synthesizer**: Integrate `tts.py` → Enhanced VoiceSynthesizer
- **Audio Reconstructor**: Implement final stage with enhanced interface

## File Structure

```
src/hyper_audio/pipeline/
├── __init__.py                    # Main exports
├── core.py                        # ResilientAudioPipeline (main orchestrator)
├── models.py                      # Data models (PipelineState, StageMetrics, etc.)
├── constants.py                   # Configuration constants and enums
├── checkpoint.py                  # CheckpointManager and StateManager
├── core_helpers.py                # Utility functions for pipeline operations
├── analytics.py                   # Full analytics with visualizations
├── analytics_simple.py            # Simplified analytics for monitoring
├── stage_interface.py             # Enhanced stage interface and type system
├── config_manager.py              # Configuration management and validation
└── stages/
    ├── __init__.py                # Stage exports
    ├── base.py                    # Base stage interfaces
    ├── preprocessor.py            # Audio preprocessing stage
    ├── enhanced_separator.py      # Enhanced voice separator + speech enhancer
    ├── diarizer.py                # Speaker diarization stage  
    ├── recognizer.py              # Speech recognition stage
    ├── synthesizer.py             # Voice synthesis stage
    └── reconstructor.py           # Audio reconstruction stage

utils/
├── model_loader.py                # Reusable model loading utilities
├── audio_utils.py                 # Audio processing utilities
└── logging_utils.py               # Logging configuration

examples/
├── multi_stage_config.yaml        # Example multi-stage configuration
└── test_enhanced_pipeline.py      # Configuration system demonstration

dev_playground.py                  # Development environment for stage testing
```

## Enhanced Usage Examples

### Basic Configuration-Based Usage

```python
from hyper_audio.pipeline.config_manager import ConfigManager
from hyper_audio.pipeline import ResilientAudioPipeline

# Initialize configuration manager
config_manager = ConfigManager()

# Register available stages
config_manager.register_stage(AudioPreprocessor, "AudioPreprocessor")
config_manager.register_stage(EnhancedVoiceSeparator, "VoiceSeparator") 
config_manager.register_stage(SpeechEnhancer, "SpeechEnhancer")

# Load and validate configuration
pipeline_config = config_manager.load_config("multi_stage_config.yaml")
is_valid, errors = config_manager.validate_config(pipeline_config)

if is_valid:
    # Get optimal execution order
    execution_order = config_manager.get_execution_order(pipeline_config)
    print(f"Execution order: {' → '.join(execution_order)}")
    
    # Initialize and run pipeline
    pipeline = ResilientAudioPipeline(
        config=pipeline_config,
        checkpoint_dir="./checkpoints",
        max_retries=3
    )
    
    result = await pipeline.process_audio_with_config(
        input_path="podcast.wav",
        output_path="enhanced_podcast.wav"
    )
```

### Multi-Stage Speech Enhancement

```python
# Example: Process audio through music separation + speech enhancement
config = {
    "stages": [
        {
            "stage_class": "AudioPreprocessor",
            "stage_id": "preprocessor",
            "config": {"target_sample_rate": 44100}
        },
        {
            "stage_class": "VoiceSeparator", 
            "stage_id": "music_separator",
            "config": {"model_name": "htdemucs_ft"}
        },
        {
            "stage_class": "SpeechEnhancer",
            "stage_id": "noise_reducer", 
            "config": {"enhancement_level": "aggressive"}
        }
    ],
    "connections": [
        {"from_stage": "preprocessor", "from_output": "audio_with_sr", 
         "to_stage": "music_separator", "to_input": "audio_input"},
        {"from_stage": "music_separator", "from_output": "vocals",
         "to_stage": "noise_reducer", "to_input": "audio_input"}
    ]
}

pipeline = ResilientAudioPipeline(config=config)
result = await pipeline.process_audio("input.wav", "output.wav")
```

### Stage Compatibility Checking

```python
# Verify stage compatibility before configuration
voice_separator = EnhancedVoiceSeparator("separator")
speech_enhancer = SpeechEnhancer("enhancer")

connections = voice_separator.can_connect_to(speech_enhancer)
if connections:
    print("Compatible connections:")
    for output, compatible_inputs in connections.items():
        print(f"  {output} → {compatible_inputs}")
```

### Development and Testing

```python
# Use development playground for individual stage testing
from dev_playground import DevPlayground

playground = DevPlayground()

# Test voice separation independently
result = playground.run_stage(
    stage="separator",
    input_data="preprocessor_12345678.pkl",
    profile=True
)

# Inspect results
playground.inspect_result("separator_12345678.pkl")
```

## Key Architectural Benefits

1. **Type Safety**: Compile-time validation prevents incompatible stage connections
2. **Flexible Configuration**: YAML/JSON-based pipeline definitions support complex workflows
3. **Dependency Validation**: Automatic detection of missing connections and circular dependencies
4. **Performance Optimization**: Multi-stage processing allows for specialized model optimization
5. **Development Efficiency**: Playground environment enables rapid iteration on individual stages
6. **Production Ready**: Enterprise-grade reliability with checkpointing, recovery, and monitoring

This enhanced architecture provides a robust foundation for production-scale audio processing with configurable workflows, type safety, and comprehensive development tools.
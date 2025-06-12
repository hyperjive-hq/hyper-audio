# Hyper Audio

## 🎯 Project Goals

This project aims to create a complete AI-powered audio processing pipeline that can transform podcasts by replacing speaker voices while maintaining natural timing, cadence, and conversation flow. All processing is done locally using open-source AI models optimized for NVIDIA RTX 4090.

### Core Objectives

1. **Voice Isolation & Separation**: Extract clean speech from podcasts with background music/noise
2. **Speaker Diarization**: Identify and separate different speakers in multi-person conversations  
3. **Speech Recognition**: Transcribe speech with speaker attribution and timing
4. **Voice Replacement**: Replace target speaker(s) with AI-generated voice while preserving:
   - Original timing and pacing
   - Natural speech patterns and inflection
   - Conversation flow and overlaps
5. **Audio Reconstruction**: Seamlessly blend replaced voice with original audio

### Technical Requirements

- **Hardware**: NVIDIA RTX 4090 (24GB VRAM)
- **Software**: All models run locally using open-source AI
- **Performance**: Real-time or near real-time processing for typical podcast lengths
- **Quality**: Broadcast-quality output indistinguishable from original

## 📋 Development Phases & TODO List

### Phase 1: Foundation & Research ⚙️
**Status**: ✅ Completed

#### ✅ Completed
- [x] Project structure and dependency management
- [x] Configuration system with environment variables
- [x] Logging and audio utility frameworks
- [x] Basic virtual environment setup
- [x] **Pipeline Orchestration**: Complete resilient pipeline with error handling and recovery
- [x] **Memory Management**: GPU memory optimization and cleanup utilities
- [x] **Checkpoint System**: Full state management and recovery capabilities
- [x] **Analytics Engine**: Comprehensive monitoring and performance tracking
- [x] **Test Suite**: Complete test coverage for pipeline infrastructure

#### 🔍 Research Required
- [ ] **Voice Cloning Models**: Research latest open-source TTS models that support voice cloning
  - Evaluate: MetaVoice-1B, Tortoise-TTS, Bark, Coqui TTS
  - **Requirements needed**: Voice quality benchmarks, inference speed, VRAM usage
- [ ] **Speaker Diarization**: Compare pyannote.audio vs alternatives (SpeakerBox, resemblyzer)
  - **Requirements needed**: Accuracy metrics for multi-speaker scenarios
- [ ] **Vocal Separation**: Evaluate Demucs alternatives (Spleeter, OpenUnmix)
  - **Requirements needed**: Quality metrics for speech vs music separation

### Phase 2: Core Audio Processing 🎵
**Status**: 🟡 In Progress - Pipeline Integration Needed

#### ✅ Standalone Scripts Available
- [x] **Vocal Separation**: `remove_music.py` - HTDemucs implementation
- [x] **Speaker Diarization**: `transcribe.py` - pyannote.audio integration
- [x] **Speech Recognition**: `transcribe.py` - Whisper integration

#### 🔄 Integration Tasks (High Priority)
- [ ] **BasePipelineStage Interface**: Create abstract base class for all stages
- [ ] **Audio Preprocessing Stage**: Implement first pipeline stage
  - [ ] Multi-format audio input support (MP3, WAV, FLAC, etc.)
  - [ ] Adaptive sample rate conversion
  - [ ] Audio normalization and cleanup
- [ ] **Voice Separator Stage**: Integrate `remove_music.py` into pipeline
  - [ ] Adapt HTDemucs code to stage interface
  - [ ] Add checkpoint serialization
- [ ] **Speaker Diarizer Stage**: Integrate `transcribe.py` diarization
  - [ ] Extract diarization logic from standalone script
  - [ ] Add speaker segment checkpoint format
- [ ] **Speech Recognizer Stage**: Integrate `transcribe.py` Whisper
  - [ ] Extract transcription logic from standalone script
  - [ ] Add transcription checkpoint format

#### 🔍 Research Required
- [ ] **Audio Quality Metrics**: Define objective measures for separation quality
- [ ] **Edge Case Handling**: Research solutions for overlapping speech, background noise
- [ ] **Performance Optimization**: CUDA optimization for audio processing pipelines

### Phase 3: Speech Recognition & Analysis 🎤
**Status**: 🔴 Not Started

#### Implementation Tasks
- [ ] **Whisper Integration**
  - [ ] Local Whisper model deployment (large-v2/v3)
  - [ ] Batch processing for long audio segments
  - [ ] Speaker-attributed transcription alignment
- [ ] **Speech Analysis**
  - [ ] Prosody and timing analysis
  - [ ] Speaking pattern extraction
  - [ ] Emotion/tone detection for voice matching

#### 🔍 Research Required
- [ ] **Whisper Alternatives**: Evaluate faster local ASR models (wav2vec2, SpeechT5)
- [ ] **Timing Precision**: Research methods for sub-word timing alignment
- [ ] **Voice Characteristics**: Define measurable voice attributes for matching

#### 🏗️ Architecture Design Needed
- [ ] **Transcription Pipeline**: Design efficient batching and memory management
- [ ] **Data Structures**: Define schemas for annotated audio segments

### Phase 4: Voice Synthesis & Cloning 🗣️
**Status**: 🟡 In Progress - Pipeline Integration Needed

#### ✅ Standalone Scripts Available
- [x] **Voice Synthesis**: `tts.py` - MetaVoice-1B implementation

#### 🔄 Integration Tasks (High Priority)
- [ ] **Voice Synthesizer Stage**: Integrate `tts.py` into pipeline
  - [ ] Adapt MetaVoice-1B code to stage interface
  - [ ] Add synthesized audio checkpoint format
- [ ] **Voice Matching System**
  - [ ] Source voice analysis and characterization
  - [ ] Target voice parameter adjustment
  - [ ] Quality validation and similarity scoring

#### 🔍 Research Required  
- [ ] **Voice Cloning Quality**: Benchmark different models for naturalness and similarity
  - **Requirements needed**: Subjective quality metrics, A/B testing framework
- [ ] **Few-Shot Learning**: Research minimum sample requirements for voice cloning
- [ ] **Real-time Synthesis**: Evaluate streaming TTS for interactive applications

#### 🏗️ Architecture Design Needed
- [ ] **Model Pipeline**: Design efficient loading/switching between voice models
- [ ] **Quality Control**: Automated quality assessment for synthetic speech

### Phase 5: Audio Reconstruction & Mixing 🎛️
**Status**: 🔴 Not Started - Needs Implementation

#### 🔄 Integration Tasks (High Priority)
- [ ] **Audio Reconstructor Stage**: Implement final pipeline stage
  - [ ] Timing preservation and alignment
  - [ ] Dynamic time warping for speech alignment
  - [ ] Pause and breath timing preservation
  - [ ] Natural speech pacing reconstruction
- [ ] **Audio Blending**
  - [ ] Seamless crossfading between original and synthetic audio
  - [ ] Background audio reintegration
  - [ ] Final mastering and quality enhancement

#### 🔍 Research Required
- [ ] **Audio Alignment**: Research best practices for speech timing alignment
- [ ] **Quality Enhancement**: Post-processing techniques for synthetic speech
- [ ] **Perceptual Quality**: Metrics for human-perceived audio quality

#### 🏗️ Architecture Design Needed
- [ ] **Real-time Processing**: Design for streaming audio reconstruction
- [ ] **Quality Pipeline**: Automated quality control and validation

### Phase 6: Integration & Optimization 🚀
**Status**: 🔴 Not Started

#### Implementation Tasks
- [ ] **End-to-End Pipeline**
  - [ ] Complete workflow orchestration
  - [ ] Error handling and recovery mechanisms
  - [ ] Progress tracking and user feedback
- [ ] **Performance Optimization**
  - [ ] GPU memory optimization for RTX 4090
  - [ ] Model quantization and optimization
  - [ ] Parallel processing where possible
- [ ] **User Interface**
  - [ ] Command-line interface with progress bars
  - [ ] Configuration file support
  - [ ] Batch processing capabilities

#### 🏗️ Architecture Design Needed
- [ ] **System Integration**: Define interfaces between all pipeline components
- [ ] **Resource Management**: Optimize for single GPU deployment
- [ ] **User Experience**: Design intuitive workflow for non-technical users

### Phase 7: Testing & Validation 🧪
**Status**: 🔴 Not Started

#### Implementation Tasks
- [ ] **Quality Assurance**
  - [ ] Automated testing suite for each component
  - [ ] Integration tests for complete pipeline
  - [ ] Performance benchmarking on standard datasets
- [ ] **User Testing**
  - [ ] Beta testing with real podcast content
  - [ ] Quality assessment by human evaluators
  - [ ] Edge case testing and bug fixes

#### 🔍 Research Required
- [ ] **Evaluation Metrics**: Define comprehensive quality metrics for voice replacement
- [ ] **Benchmark Datasets**: Identify or create standard test datasets
- [ ] **Success Criteria**: Define measurable goals for project completion

## 🛠️ Technical Stack

### Core AI Models (Local/Open Source)
- **Speech Recognition**: OpenAI Whisper (large-v2/v3)
- **Speaker Diarization**: pyannote.audio
- **Vocal Separation**: Meta Demucs
- **Voice Synthesis**: MetaVoice-1B / Tortoise-TTS / Bark
- **Speech Processing**: SpeechBrain ecosystem

### Infrastructure
- **GPU**: NVIDIA RTX 4090 (24GB VRAM)
- **Framework**: PyTorch with CUDA optimization
- **Audio**: librosa, soundfile, pydub
- **ML Pipeline**: HuggingFace Transformers

## 📊 Success Metrics

### Technical Metrics
- **Processing Speed**: < 2x real-time for typical podcast length
- **Voice Quality**: > 4.0/5.0 subjective quality score
- **Speaker Accuracy**: > 95% diarization accuracy
- **Timing Preservation**: < 50ms deviation from original timing

### User Experience Metrics
- **Ease of Use**: Single command processing
- **Reliability**: < 5% failure rate on diverse content
- **Resource Efficiency**: < 20GB VRAM peak usage

## 🚧 Current Development Focus

**🔥 IMMEDIATE NEXT STEPS (High Priority):**
1. **Create BasePipelineStage Interface** - Define common interface for all 6 stages
2. **Integrate Existing Scripts** - Convert standalone scripts to pipeline stages:
   - `remove_music.py` → VoiceSeparator stage
   - `transcribe.py` → SpeakerDiarizer + SpeechRecognizer stages  
   - `tts.py` → VoiceSynthesizer stage
3. **Implement Missing Stages**:
   - AudioPreprocessor (first stage)
   - AudioReconstructor (final stage)
4. **End-to-End Testing** - Complete pipeline with all 6 stages

**Pipeline Infrastructure Status: ✅ COMPLETE**
- Resilient pipeline orchestration with checkpointing
- State management and recovery
- Analytics and monitoring
- Comprehensive test coverage

**Areas Needing Requirements:**
- Voice quality benchmarking methodology
- User interface design specifications
- Performance optimization targets

**Research Questions:**
- Best open-source voice cloning for real-time use?
- Optimal model quantization strategies for RTX 4090?
- Edge case handling for complex audio scenarios?

## 🛠️ Development Tools

### Pipeline Stage Development Playground

For developing and testing individual pipeline stages, use the included development playground:

```bash
# Make the playground executable
chmod +x dev_playground.py

# List available stages and their descriptions
python dev_playground.py list

# List available sample data
python dev_playground.py list --stage any

# Run a specific stage with profiling
python dev_playground.py run --stage preprocessor --input data/sample.mp3

# Run a stage with cached intermediate results
python dev_playground.py run --stage diarizer --input preprocessor_1234567890

# Profile a stage's performance
python dev_playground.py profile --stage separator --input preprocessor_result.pkl

# Inspect cached results or data files
python dev_playground.py inspect --input diarizer_output.pkl
```

#### Playground Features

- **Individual Stage Testing**: Run any pipeline stage independently
- **Automatic Input Discovery**: Finds appropriate sample data or cached results
- **Performance Profiling**: Memory usage, execution time, and result analysis
- **Result Caching**: Save intermediate results for rapid iteration
- **Hot Reloading**: Test code changes without full pipeline runs

#### Development Workflow

1. **Prepare Sample Data**: Place audio files in `data/` directory
2. **Run Early Stages**: Generate intermediate results for later stages
   ```bash
   python dev_playground.py run --stage preprocessor --input data/podcast.wav
   python dev_playground.py run --stage separator --input preprocessor_1234567890
   ```
3. **Iterate on Target Stage**: Modify stage code and test quickly
   ```bash
   python dev_playground.py run --stage diarizer --input separator_1234567890
   ```
4. **Profile Performance**: Optimize memory and speed
   ```bash
   python dev_playground.py profile --stage diarizer
   ```

#### Cached Results

The playground stores intermediate results in `dev_cache/` with automatic naming:
- `preprocessor_<timestamp>.pkl` - Normalized audio data
- `separator_<timestamp>.pkl` - Separated audio components  
- `diarizer_<timestamp>.pkl` - Speaker segments
- `recognizer_<timestamp>.pkl` - Transcription with timing
- `synthesizer_<timestamp>.pkl` - Generated speech
- `reconstructor_<timestamp>.pkl` - Final mixed audio

This allows rapid iteration on individual stages without re-running the entire pipeline.

### Pipeline Stage Reference

Each stage has specific input/output requirements. Here are detailed examples for each:

#### 1. AudioPreprocessor Stage
**Purpose**: Load, normalize, and prepare raw audio files for processing

```bash
# Input: Raw audio file (MP3, WAV, FLAC, etc.)
python dev_playground.py run --stage preprocessor --input data/podcast.wav

# Expected Input: File path (string)
# Expected Output: Tuple of (normalized_audio: np.ndarray, sample_rate: int)
```

**Input**: Audio file path (`.wav`, `.mp3`, `.flac`, `.m4a`)
**Output**: `(audio_data, sample_rate)` where:
- `audio_data`: Normalized numpy array of shape `(samples,)` or `(channels, samples)`
- `sample_rate`: Integer sample rate (typically 16000 or 44100 Hz)

#### 2. VoiceSeparator Stage  
**Purpose**: Separate vocals from background music using AI models

```bash
# Input: Preprocessed audio data
python dev_playground.py run --stage separator --input preprocessor_1234567890

# Expected Input: (audio_data: np.ndarray, sample_rate: int)
# Expected Output: Dict with 'vocals' and 'music' audio arrays
```

**Input**: `(audio_data, sample_rate)` from preprocessor
**Output**: Dictionary with keys:
- `"vocals"`: Separated vocal audio as numpy array
- `"music"`: Separated background music as numpy array

#### 3. SpeakerDiarizer Stage
**Purpose**: Identify different speakers and their timing segments

```bash
# Input: Separated vocal audio
python dev_playground.py run --stage diarizer --input separator_1234567890

# Expected Input: (vocals_audio: np.ndarray, sample_rate: int)  
# Expected Output: List of speaker segments with timing
```

**Input**: `(vocals_audio, sample_rate)` from separator
**Output**: List of speaker segments:
```python
[
    {
        "speaker": "Speaker_A",
        "start": 0.0,        # Start time in seconds
        "end": 5.2,          # End time in seconds  
        "confidence": 0.95   # Confidence score
    },
    # ... more segments
]
```

#### 4. SpeechRecognizer Stage
**Purpose**: Transcribe speech with speaker attribution and precise timing

```bash
# Input: Vocal audio + speaker segments
python dev_playground.py run --stage recognizer --input diarizer_1234567890

# For manual input, you need both vocals and segments:
# The playground will automatically load the right data combination
```

**Input**: `(vocals_audio, sample_rate, speaker_segments)`
**Output**: Transcription dictionary:
```python
{
    "full_text": "Complete transcription...",
    "segments": [
        {
            "speaker": "Speaker_A",
            "text": "Hello, how are you?",
            "start": 0.0,
            "end": 2.5,
            "confidence": 0.92
        },
        # ... more segments
    ]
}
```

#### 5. VoiceSynthesizer Stage
**Purpose**: Generate replacement voice for target speaker

```bash
# Input: Transcription data
python dev_playground.py run --stage synthesizer --input recognizer_1234567890

# With custom target speaker and voice
python dev_playground.py run --stage synthesizer --input recognizer_1234567890 \
    --target-speaker "Speaker_A" --replacement-voice "path/to/voice.wav"
```

**Input**: `transcription` dict from recognizer + optional parameters
**Output**: Dictionary mapping speakers to audio:
```python
{
    "Speaker_A": np.ndarray,  # Synthesized audio for replaced speaker
    "timing_info": [          # Timing alignment data
        {"start": 0.0, "end": 2.5, "audio_start": 0, "audio_end": 40000},
        # ... more timing info
    ]
}
```

#### 6. AudioReconstructor Stage
**Purpose**: Combine synthesized voice with original music to create final output

```bash
# Input: All previous stage outputs combined
python dev_playground.py run --stage reconstructor --input synthesizer_1234567890

# The reconstructor automatically loads required data from previous stages
```

**Input**: Multiple components:
- `separated_audio`: From separator stage
- `synthesized_audio`: From synthesizer stage  
- `transcription`: From recognizer stage
- `sample_rate`: Audio sample rate

**Output**: Final reconstructed audio as numpy array ready for export

### Common Development Patterns

#### Testing Stage Chain
```bash
# Run stages in sequence, each using output from previous
python dev_playground.py run --stage preprocessor --input data/test.wav
python dev_playground.py run --stage separator --input preprocessor_1234567890
python dev_playground.py run --stage diarizer --input separator_1234567890
python dev_playground.py run --stage recognizer  # Auto-finds diarizer output
python dev_playground.py run --stage synthesizer # Auto-finds recognizer output  
python dev_playground.py run --stage reconstructor # Auto-finds all inputs
```

#### Performance Optimization
```bash
# Profile each stage to identify bottlenecks
python dev_playground.py profile --stage preprocessor
python dev_playground.py profile --stage separator --input preprocessor_result
python dev_playground.py profile --stage diarizer --input separator_result
```

#### Result Inspection
```bash
# Examine intermediate results
python dev_playground.py inspect --input preprocessor_1234567890
python dev_playground.py inspect --input diarizer_output.pkl
python dev_playground.py inspect --input transcription.pkl
```

#### Debugging Failed Stages
```bash
# Run with verbose logging and profiling
python dev_playground.py run --stage diarizer --input vocals.pkl --profile

# Inspect inputs to understand format issues
python dev_playground.py inspect --input vocals.pkl
```

### Sample Data Management

Place sample files in the `data/` directory:
```
data/
├── podcast_sample.wav      # Short podcast clip
├── speech_sample.mp3       # Single speaker sample  
├── conversation.wav        # Multi-speaker sample
└── reference_voice.wav     # Voice for synthesis
```

The playground automatically:
- Discovers appropriate sample data for each stage
- Caches intermediate results with timestamps
- Manages dependencies between stages
- Provides detailed profiling and inspection tools

This development environment allows you to refine each stage independently before integration into the full pipeline.

## 🤝 Contributing

This is an experimental project exploring the boundaries of local AI audio processing. Areas particularly needing research and development are marked with 🔍 and 🏗️ above.

## 📄 License

MIT License - See LICENSE file for details



  1. Asteroid Models (SpeechBrain ecosystem)

  "speechbrain/sepformer-wham"           # Speech + noise + reverb
  "speechbrain/sepformer-whamr"          # + room acoustics
  "speechbrain/sepformer-wsj02mix"       # Multi-speaker + noise
  "speechbrain/dualpath-rnn-wsj0-2mix"   # Classic multi-source

  2. ONNX Speech Enhancement Models

  "microsoft/speechtokenizer-base"        # Microsoft's multi-domain
  "nvidia/speechtokenizer"               # NVIDIA's approach

  3. Universal Source Separation

  "asteroid/ConvTasNet_WHAM"             # 3-source: speech + 2 noise types
  "asteroid/DPRNNTasNet_WHAMR"           # + reverberation handling
  "facebook/bandit-v1_0"                 # Facebook's universal separator

  4. Research Models (Cutting Edge)

  "speechbrain/resepformer-wsj02mix"     # Recent state-of-the-art
  "kaituoxu/Conv-TasNet"                 # Lightweight option
  "asteroid/DPRNN_TAC"                   # Time-domain separation

  5. Commercial-Grade Options

  "crisp/speech-enhancement"             # Multi-modal enhancement
  "elevenlabs/speech-isolator"           # If available

  🏆 Top Recommendations

  1. speechbrain/sepformer-whamr - Best all-around for speech + music + noise + reverb
  2. asteroid/DPRNNTasNet_WHAMR - Good performance/speed balance
  3. facebook/bandit-v1_0 - If you can access it (universal separator)
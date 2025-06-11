# Installation Guide

## Quick Start

### Option 1: Basic Installation (Recommended)
```bash
# Install core dependencies only
pip install -e .

# Install development tools
pip install -e ".[dev]"

# Run tests
pytest
```

### Option 2: Full Installation with ML Packages
```bash
# Install all dependencies (may have compatibility issues on Python 3.13+)
pip install -e ".[all]"
```

## Troubleshooting

### Python 3.13 Compatibility Issues

If you're using Python 3.13 and encounter installation errors with certain packages:

#### 1. OpenAI Whisper Installation Issues
```bash
# If openai-whisper fails, try installing from git
pip install git+https://github.com/openai/whisper.git

# Or use the transformers-based alternative
pip install transformers[torch]
```

#### 2. BitsAndBytesConfig Issues
The `bitsandbytes` package may not work with Python 3.13. This is optional for quantization:
```bash
# Install without quantization support
pip install -e . --no-deps
pip install torch transformers huggingface-hub accelerate
pip install librosa soundfile pydub scipy
pip install speechbrain demucs pyannote.audio
pip install numpy tqdm python-dotenv setuptools
```

#### 3. PyTorch Installation
For specific CUDA versions:
```bash
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# CPU only
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Alternative Whisper Installation

If `openai-whisper` fails, you can use the HuggingFace transformers version:

```python
# In your code, replace whisper usage with:
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

processor = AutoProcessor.from_pretrained("openai/whisper-large-v2")
model = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v2")
```

## Dependencies by Category

### Core Pipeline (Always Required)
- `torch` - PyTorch for ML models
- `transformers` - HuggingFace models
- `numpy` - Numerical computing
- `librosa` - Audio processing
- `soundfile` - Audio I/O
- `tqdm` - Progress bars

### Audio Processing Models
- `speechbrain` - Speech processing toolkit
- `demucs` - Music source separation
- `pyannote.audio` - Speaker diarization

### Optional ML Enhancements
- `bitsandbytes` - Model quantization (Python < 3.13 only)
- `accelerate` - HuggingFace acceleration

### Development Tools
- `pytest` - Testing framework
- `black` - Code formatting
- `isort` - Import sorting
- `mypy` - Type checking

## Verification

Test your installation:

```bash
# Run basic tests
pytest tests/test_models.py -v

# Test pipeline components
pytest tests/test_checkpoint.py -v

# Check imports
python -c "from src.hyper_audio.pipeline import ResilientAudioPipeline; print('✓ Pipeline imported successfully')"
```

## Docker Alternative

If you continue having issues, consider using Docker:

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY . .

RUN pip install -e ".[all]"

CMD ["pytest"]
```

## Support

If you encounter issues:

1. **Check Python version**: `python --version` (3.8-3.12 recommended)
2. **Update pip**: `pip install --upgrade pip setuptools wheel`
3. **Clear cache**: `pip cache purge`
4. **Virtual environment**: Always use a virtual environment
5. **System dependencies**: Ensure you have build tools installed

### System Dependencies

#### Ubuntu/Debian
```bash
sudo apt update
sudo apt install build-essential python3-dev libsndfile1-dev ffmpeg
```

#### macOS
```bash
brew install ffmpeg libsndfile
```

#### Windows
```bash
# Install Visual Studio Build Tools
# Install ffmpeg from https://ffmpeg.org/download.html
```
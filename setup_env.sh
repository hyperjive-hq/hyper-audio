#!/bin/bash

# AI Audio Pipeline Environment Setup Script

set -e  # Exit on any error

echo "=== AI Audio Pipeline Environment Setup ==="

# Check if virtual environment exists
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python -m venv .venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install the package in development mode
echo "Installing ai-audio-pipeline in development mode..."
pip install -e .

# Install development dependencies
echo "Installing development dependencies..."
pip install -e .[dev,test,docs]

# Install PyTorch with CUDA support for NVIDIA 4090
echo "Installing PyTorch with CUDA support..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Create .env file template if it doesn't exist
if [ ! -f ".env" ]; then
    echo "Creating .env template..."
    cat > .env << 'EOF'
# Hugging Face token for model downloads
HUGGINGFACE_TOKEN=your_token_here

# Device configuration
TORCH_DEVICE=cuda
CUDA_VISIBLE_DEVICES=0

# Model cache directory
MODEL_CACHE_DIR=~/.cache/ai_audio_pipeline

# Logging configuration
LOG_LEVEL=INFO
LOG_FILE=logs/ai_audio_pipeline.log

# Audio processing settings
DEFAULT_SAMPLE_RATE=16000
MAX_AUDIO_LENGTH=3600  # seconds

# Model configurations
WHISPER_MODEL=large-v2
DEMUCS_MODEL=htdemucs
DIARIZATION_MODEL=pyannote/speaker-diarization
TTS_MODEL=metavoiceio/metavoice-1B-v0.1
EOF
    echo "Created .env template - please update with your tokens"
fi

# Create directories
echo "Creating project directories..."
mkdir -p logs
mkdir -p data/input
mkdir -p data/output
mkdir -p data/cache

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "Next steps:"
echo "1. Update .env file with your Hugging Face token"
echo "2. Activate the environment: source .venv/bin/activate"
echo "3. Test installation: python -c 'import torch; print(torch.cuda.is_available())'"
echo ""
echo "Project structure created with:"
echo "- pyproject.toml for dependency management"
echo "- Virtual environment in .venv/"
echo "- Environment variables in .env"
echo "- Directory structure for data and logs"
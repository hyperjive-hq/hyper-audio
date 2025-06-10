import argparse

import librosa
import torch
import scipy.io.wavfile
import logging
from transformers import AutoProcessor, AutoModel
import soundfile as sf

CUDA = "cuda"

# Set up logging for better debugging and user feedback
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
models = {}

"""
Example Usage:
    python script.py --text_file input.txt --sample_audio sample.wav --output speech.wav --hf_token your_hf_token

    - '--text_file input.txt': Path to a text file containing the text to convert to speech (e.g., "Hello, this is a test.").
    - '--sample_audio sample.wav': Path to a ~30-second WAV file of the voice to clone (e.g., a recording of someone speaking).
    - '--output speech.wav': Path where the generated speech WAV file will be saved (default: 'output.wav').
    - '--hf_token your_hf_token': Optional Hugging Face token for model download authentication (e.g., 'hf_xxxxxxxxxxxxxxxxxxxxxx').

    Prerequisites:
    - Install required packages: pip install torch scipy pydub huggingface_hub
    - Manually download and set up 'metavoice-src' from https://github.com/metavoiceio/metavoice-src in the working directory.
    - Ensure NVIDIA 4090 GPU with CUDA and ffmpeg are installed.
"""

def parse_args():
    """
    Parse command-line arguments using argparse to handle user inputs.

    Returns:
        argparse.Namespace: Parsed arguments containing input file paths and optional HF token.

    Example:
        Command: python script.py --text_file input.txt --sample_audio sample.wav
        Result: args.text_file = 'input.txt', args.sample_audio = 'sample.wav', args.output = 'output.wav', args.hf_token = None
    """
    parser = argparse.ArgumentParser(description="Convert text to speech using MetaVoice-1B on NVIDIA 4090.")
    parser.add_argument('--text_file', type=str, required=True, 
                        help="Path to the input text file to convert to speech.")
    parser.add_argument('--sample_audio', type=str, required=True, 
                        help="Path to the sample audio file for voice cloning.")
    parser.add_argument('--output', type=str, default="output.wav", 
                        help="Path to save the output WAV file (default: output.wav).")
    parser.add_argument('--hf_token', type=str, default=None, 
                        help="Optional Hugging Face token to authenticate model download.")
    return parser.parse_args()

def load_model(hf_token=None):
    repo_id = "metavoiceio/metavoice-1B-v0.1"
    models["metavoice_processor"] = AutoProcessor.from_pretrained(repo_id, use_auth_token=hf_token)
    models["metavoice_model"] = AutoModel.from_pretrained(repo_id).to(CUDA)

# Preprocess audio with librosa in memory
def preprocess_audio_in_memory(input_path, target_sr=22050, duration=30):
    # Load audio with librosa
    audio, sr = librosa.load(input_path, sr=None, mono=False)  # Load with original samplerate
    
    # Convert to mono if stereo
    if len(audio.shape) > 1:
        audio = librosa.to_mono(audio)
    
    # Resample to target samplerate
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    
    # Trim silence (top_db=20 is a reasonable threshold for voice)
    audio, _ = librosa.effects.trim(audio, top_db=20)
    
    # Limit duration (e.g., 30 seconds for zero-shot cloning)
    max_samples = int(duration * target_sr)
    if len(audio) > max_samples:
        audio = audio[:max_samples]
    
    # Normalize amplitude to [-1, 1]
    audio = librosa.util.normalize(audio)
    
    return audio

def text_to_speech(text_file, sample_audio, output_path, hf_token=None):
    """
    Convert text to speech using MetaVoice-1B model, leveraging the NVIDIA 4090 GPU.

    Args:
        text_file (str): Path to the input text file.
        sample_audio (str): Path to the preprocessed sample audio file for cloning.
        output_path (str): Path to save the output WAV file.
        hf_token (str, optional): Hugging Face token (included for potential future use).

    Example:
        Input: text_file='input.txt' ("Hello world"), sample_audio='sample.wav', output_path='speech.wav'
        Output: 'speech.wav' with cloned voice saying "Hello world"
        Logs: "Synthesizing speech..." followed by "Speech saved to speech.wav"
    """
    logging.info("Starting text-to-speech conversion...")

    # Ensure GPU is available (NVIDIA 4090 with 24GB VRAM)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. This script requires an NVIDIA GPU.")
    device = torch.device(CUDA)
    logging.info(f"Using GPU: {torch.cuda.get_device_name(0)}")

    # Read text from file
    with open(text_file, 'r', encoding='utf-8') as f:
        text = f.read().strip()

    # Import MetaVoice TTS module dynamically
    try:
        from fam.llm.fast_inference import TTS
    except ImportError:
        logging.error("Failed to import MetaVoice TTS module. Ensure 'metavoice-src' is correctly set up and dependencies are installed.")
        raise

    processor = models["metavoice_processor"]
    model = models["metavoice_model"]
    preprocessed_audio = preprocess_audio_in_memory(sample_audio)

    # Process the text and preprocessed audio directly
    inputs = processor(
        text=text,
        audio=preprocessed_audio,  # Pass NumPy array directly
    ).to(CUDA)

    # Generate audio with voice cloning
    with torch.no_grad():
        audio_output = model.generate(
            **inputs,
            do_sample=True,  # Optional: for naturalness
            temperature=0.7  # Optional: adjust randomness
        )

        # Save the cloned audio to a file
        output_file = "cloned_output_audio.wav"
        sf.write(output_file, audio_output, samplerate=target_sr)

def main():
    """
    Main function to orchestrate the text-to-speech process by calling other functions in sequence.

    Example:
        Runs the full pipeline: parse args, download model, preprocess audio, and synthesize speech.
    """
    # Parse command-line arguments
    args = parse_args()

    # Download the model files from Hugging Face if not already present
    load_model(args.hf_token)

    # Preprocess the sample audio
    processed_audio = preprocess_audio(args.sample_audio)

    # Perform text-to-speech conversion
    text_to_speech(args.text_file, processed_audio, args.output, args.hf_token)

if __name__ == "__main__":
    main()
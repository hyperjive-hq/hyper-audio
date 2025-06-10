import os
import torch
import whisper
from pyannote.audio import Pipeline
from huggingface_hub import login
import librosa
import soundfile as sf
import numpy as np
from pathlib import Path
import argparse

# Monkey-patch torch.load to use weights_only=False globally
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs.setdefault('weights_only', False)
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

def check_and_download_models(hf_token):
    """Check if models exist and download if necessary"""
    login(hf_token)
    model_dir = Path.home() / ".cache" / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    whisper_path = model_dir / "whisper-large-v2"
    if not whisper_path.exists():
        print("Downloading Whisper large-v2 model...")
        whisper.load_model("large-v2", device="cuda", download_root=str(model_dir))
    
    pyannote_path = model_dir / "pyannote" / "speaker-diarization"
    if not pyannote_path.exists():
        print("Downloading pyannote speaker diarization model...")
        Pipeline.from_pretrained("pyannote/speaker-diarization",
                               use_auth_token=hf_token)

def process_audio_file(input_file):
    """Process input audio file to compatible format"""
    supported_formats = ['.wav', '.mp3', '.flac', '.ogg']
    file_ext = os.path.splitext(input_file)[1].lower()
    if file_ext not in supported_formats:
        raise ValueError(f"Unsupported file format. Supported formats: {supported_formats}")
    
    audio, sr = librosa.load(input_file, sr=16000, mono=True)
    if file_ext != '.wav':
        temp_file = "temp_audio.wav"
        sf.write(temp_file, audio, sr)
        return temp_file
    return input_file

def diarize_audio(input_file, hf_token, device):
    """Diarize audio and return speaker segments"""
    print("Loading pyannote pipeline...")
    diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization",
                                                  use_auth_token=hf_token).to(device)
    
    audio, sr = librosa.load(input_file, sr=16000, mono=True)
    
    print("Performing speaker diarization...")
    diarization = diarization_pipeline({"waveform": torch.FloatTensor(audio).unsqueeze(0).to(device),
                                      "sample_rate": sr})
    
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "audio": audio[int(turn.start * sr):int(turn.end * sr)]
        })
    
    unique_speakers = len(set(segment["speaker"] for segment in segments))
    print(f"Detected {unique_speakers} unique speakers")
    
    return segments, sr

def transcribe_speakers(segments, device):
    """Transcribe audio segments for each speaker and save to files"""
    print("Loading Whisper model...")
    whisper_model = whisper.load_model("large-v2", device="cuda")
    
    speaker_transcripts = {}
    for segment in segments:
        speaker_id = segment["speaker"]
        if speaker_id not in speaker_transcripts:
            speaker_transcripts[speaker_id] = []
        
        print(f"Transcribing segment for {speaker_id}...")
        result = whisper_model.transcribe(segment["audio"],
                                        language="en",
                                        fp16=True)
        speaker_transcripts[speaker_id].append(result["text"])
    
    speaker_mapping = {spk: f"speaker{i+1}" for i, spk in enumerate(speaker_transcripts.keys())}
    for speaker_id, transcript in speaker_transcripts.items():
        output_file = f"{speaker_mapping[speaker_id]}.txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write("\n".join(transcript))
        print(f"Saved transcript to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Diarize and transcribe audio files using Whisper and pyannote")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token for model access")
    parser.add_argument("--audio-file", required=True, help="Path to the input audio file")
    
    args = parser.parse_args()
    
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. GPU required.")
    device = torch.device("cuda:0")  # NVIDIA 4090
    torch.cuda.set_device(device)
    
    if not os.path.exists(args.audio_file):
        raise FileNotFoundError(f"Audio file not found: {args.audio_file}")
    
    check_and_download_models(args.hf_token)
    processed_file = process_audio_file(args.audio_file)
    
    try:
        segments, sr = diarize_audio(processed_file, args.hf_token, device)
        transcribe_speakers(segments, device)
    finally:
        if processed_file != args.audio_file and os.path.exists(processed_file):
            os.remove(processed_file)

if __name__ == "__main__":
    # Usage: python script.py --hf-token YOUR_TOKEN --audio-file path/to/audio.wav
    try:
        main()
    except Exception as e:
        print(f"Error: {str(e)}")
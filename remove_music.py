import argparse
import logging
import os
import sys
from typing import Tuple
import soundfile as sf
import librosa
import numpy as np
import torch
from demucs import pretrained
from demucs.apply import apply_model
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("voice_demux")

DEMUCS_MODEL = "htdemucs"

class RemoveMusic:
    def __init__(self, device: str = "cuda", hf_token: str = None):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.hf_token = hf_token
        self.target_sample_rate = 44100  # HTDemucs expects 44.1 kHz
        self.models = {}
        logger.info(f"Using device: {self.device}, Target Sample Rate: {self.target_sample_rate}Hz")
        self.load_models()  # Load models on initialization

    def load_models(self):
        """Load the HTDemucs model for vocal separation."""
        logger.info("Loading HTDemucs model for vocal separation...")
        self.models["demucs"] = pretrained.get_model(name="htdemucs").to(self.device)
        logger.info("HTDemucs model loaded successfully")
    
    def load_and_prepare_audio(self, file_path: str) -> np.ndarray:
        """Load audio file, resample to 44.1 kHz if necessary, and ensure 2 channels (stereo)."""
        logger.info(f"Loading audio file: {file_path}")
        audio, sr = librosa.load(file_path, sr=None, mono=False)  # Load with native sample rate, preserve stereo/mono

        # Check sample rate
        logger.info(f"Original sample rate: {sr}Hz")
        if sr != self.target_sample_rate:
            logger.info(f"Resampling audio from {sr}Hz to {self.target_sample_rate}Hz for HTDemucs compatibility")
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.target_sample_rate)
            sr = self.target_sample_rate
        else:
            logger.info("Sample rate already matches target, no resampling needed")

        # Ensure audio is 2D: [channels, time]
        if audio.ndim == 1:
            logger.info("Detected mono audio (1 channel), converting to stereo (2 channels) for Demucs")
            audio = np.stack([audio, audio], axis=0)  # Duplicate mono channel to create stereo: [2, time]
        elif audio.ndim == 2:
            if audio.shape[0] == 1:
                logger.info("Detected mono audio (1 channel), converting to stereo (2 channels) for Demucs")
                audio = np.vstack([audio, audio])  # Duplicate mono channel to create stereo: [2, time]
            elif audio.shape[0] == 2:
                logger.info("Detected stereo audio (2 channels), no conversion needed")
            elif audio.shape[0] > 2:
                raise ValueError("Audio must have 1 (mono) or 2 (stereo) channels; more than 2 channels detected")

        logger.info(f"Prepared audio shape: {audio.shape}, Sample rate: {sr}Hz")
        return audio

    def separate_vocals(self, file_path: str) -> Tuple[np.ndarray, np.ndarray]:
        """Separate vocals from background using HTDemucs from an audio file.

        Args:
            file_path: Path to the audio file.

        Returns:
            Tuple of (vocals, background) as numpy arrays.
        """
        if "demucs" not in self.models:
            raise RuntimeError("Demucs model not loaded. Call load_models() first.")

        # Load and prepare audio
        audio = self.load_and_prepare_audio(file_path)

        logger.info("Starting vocal separation...")
        # Convert to tensor
        audio_tensor = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).to(self.device)  # [1, channels, time]

        # Process with progress bar
        with tqdm(total=100, desc="Vocal separation", unit="%") as pbar:
            pbar.update(10)  # Setup
            with torch.no_grad():
                logger.info("Running HTDemucs model (this may take several minutes for long audio)...")
                sources = apply_model(self.models["demucs"], audio_tensor, device=self.device)  # Use apply_model instead of direct call
                pbar.update(70)  # Inference

            # Extract vocals and background
            vocals = sources[0, 3].cpu().numpy()  # Vocals (4th stem)
            background = sources[0, :3].sum(dim=0).cpu().numpy()  # Sum drums, bass, other
            pbar.update(20)  # Post-processing

            # Clean up
            del sources, audio_tensor
            torch.cuda.empty_cache()

        duration = vocals.shape[-1] / self.target_sample_rate
        logger.info(f"Vocal separation completed. Extracted {duration:.2f} seconds of vocals")
        return vocals, background

def main():

    parser = argparse.ArgumentParser(description="Music Demux Tool")
    parser.add_argument("audio", help="Path to input audio file")
    parser.add_argument("output", help="Path to save output audio")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token for model access")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Processing device")
    parser.add_argument("--verbose", action="store_true", help="Enable more detailed logging")

    args = parser.parse_args()
    if args.verbose:
        logging.getLogger("voice_demux").setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")

    try:
        logger.info(f"Initializing RemoveMusic with {args.device} device")
        processor = RemoveMusic(device=args.device, hf_token=args.hf_token)

        logger.info("Loading required models...")
        processor.load_models()

        vocals, background = processor.separate_vocals(args.audio)
        # Write output files
        vocals_path = os.path.join(args.output, 'vocals.wav')
        background_path = os.path.join(args.output, 'background.wav')

        logger.info(f"Writing vocals to {vocals_path}, shape after transpose: {vocals.T.shape}")
        sf.write(vocals_path, vocals.T, processor.target_sample_rate)
        logger.info(f"Writing background to {background_path}, shape after transpose: {background.T.shape}")
        sf.write(background_path, background.T, processor.target_sample_rate)
        logger.info("Output files written successfully")
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()

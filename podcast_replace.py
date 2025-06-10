#!/usr/bin/env python3
"""
Podcast Voice Replacement Script

This script replaces the host's voice in a podcast audio file with a synthesized voice based on a reference sample.

Requirements:
- NVIDIA GPU with CUDA support (4090 or compatible)
- 16GB+ RAM recommended
- 12GB+ available disk space for model storage
- Python 3.8 or higher
- Required libraries: torch, transformers, huggingface_hub, librosa, soundfile, pydub, demucs, speechbrain, pyannote.audio

Usage:
python script.py podcast.mp3 reference_voice.wav output.mp3 --host-samples host_sample1.wav host_sample2.wav --hf-token your_hf_token
or
python script.py podcast.mp3 reference_voice.wav output.mp3 --host-timestamps timestamps1.json timestamps2.json --hf-token your_hf_token

Input formats: .mp3, .wav (other formats may work but are untested)
Output format: .mp3 or .wav based on output path extension

Note: This script uses hypothetical methods for MetaVoice-1B. Verify and adjust according to the actual MetaVoice-1B API.
"""
import builtins
import os
import sys
import argparse
import torch
import numpy as np
import logging
import tempfile
from typing import Dict, List, Tuple, Optional
import json
import time
from tqdm import tqdm
from pydub import AudioSegment
import soundfile as sf
import librosa
from transformers import AutoProcessor, AutoModel

""" ML Models used in this script. Will be downloaded automatically if not available from huggingface.co"""
DEMUCS_MODEL = "htdemucs"
SPEAKER_RECOGNITION_MODEL = "speechbrain/spkrec-ecapa-voxceleb"
DIARIZATION_MODEL = "pyannote/speaker-diarization"
WHISPER_MODEL = "openai/whisper-large-v2"
METAVOICE_MODEL = "metavoiceio/metavoice-1B-v0.1"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("podcast_voice_replacement")

class AudioProcessor:
    """Main class for podcast voice replacement processing"""
    
    def __init__(self, use_fp16: bool = True, device: str = "cuda", hf_token: str = None):
        """Initialize the audio processor with a fixed 16 kHz sample rate."""
        self.device = device if torch.cuda.is_available() else "cpu"
        self.use_fp16 = use_fp16 and self.device == "cuda"
        self.hf_token = hf_token
        self.sample_rate = 16000  # Fixed sample rate for all processing
        self.models = {}
        logger.info(f"Using device: {self.device}, FP16: {self.use_fp16}, Sample Rate: {self.sample_rate}Hz")
    
    def load_models(self):
        """Load required models with error handling."""
        logger.info("Loading models...")
        try:
            # Whisper for transcription
            logger.info("Loading Whisper model for transcription...")
            from transformers import WhisperProcessor, WhisperForConditionalGeneration
            self.models["whisper_processor"] = WhisperProcessor.from_pretrained(WHISPER_MODEL)
            self.models["whisper_model"] = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL).to(self.device)
            if self.use_fp16:
                self.models["whisper_model"] = self.models["whisper_model"].half()
            logger.info("Whisper model loaded successfully")
            
            # Pyannote for diarization
            logger.info("Loading Pyannote model for speaker diarization...")
            from pyannote.audio import Pipeline
            self.models["diarization"] = Pipeline.from_pretrained(DIARIZATION_MODEL, use_auth_token=self.hf_token)
            logger.info("Pyannote model loaded successfully")
            
            # SpeechBrain for speaker embedding
            logger.info("Loading SpeechBrain model for speaker embedding...")
            from speechbrain.pretrained.interfaces import EncoderClassifier
            self.models["speaker_embedding"] = EncoderClassifier.from_hparams(source=SPEAKER_RECOGNITION_MODEL)
            logger.info("SpeechBrain model loaded successfully")
            
            # MetaVoice for synthesis tts
            logger.info("Loading MetaVoice model for voice synthesis...")
            self.models["metavoice_processor"] = AutoProcessor.from_pretrained(METAVOICE_MODEL, use_auth_token=self.hf_token)
            self.models["metavoice_model"] = AutoModel.from_pretrained(METAVOICE_MODEL).to(self.device)
            if self.use_fp16:
                self.models["metavoice_model"] = self.models["metavoice_model"].half()
            logger.info("MetaVoice model loaded successfully")
            
            # Demucs for vocal separation
            logger.info("Loading Demucs model for vocal separation...")
            import demucs.pretrained
            self.models["demucs"] = demucs.pretrained.get_model(DEMUCS_MODEL).to(self.device)
            if self.use_fp16:
                self.models["demucs"] = self.models["demucs"].half()
            logger.info("Demucs model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            raise
        logger.info("All models loaded successfully")
    
    def load_audio(self, audio_path: str) -> Tuple[np.ndarray, int]:
        """Load audio file, convert to mono, and resample to 16 kHz."""
        logger.info(f"Loading audio: {audio_path}")
        try:
            audio, _ = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            duration = len(audio) / self.sample_rate
            logger.info(f"Audio loaded successfully: {duration:.2f} seconds at {self.sample_rate}Hz")
            return audio, self.sample_rate
        except Exception as e:
            logger.error(f"Failed to load audio file {audio_path}: {e}")
            raise
    
    def separate_vocals(self, audio: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Separate vocals from background using Demucs."""
        logger.info("Starting vocal separation...")
        # Create a progress bar placeholder for this operation
        with tqdm(total=100, desc="Vocal separation", unit="%") as pbar:
            audio_tensor = torch.tensor(audio).unsqueeze(0).unsqueeze(0).to(self.device)  # [1, 1, time]
            if self.use_fp16:
                audio_tensor = audio_tensor.half()
            pbar.update(25)  # Update progress after loading audio to tensor
            
            with torch.no_grad():
                logger.info("Running Demucs model (this may take several minutes for long audio)...")
                sources = self.models["demucs"](audio_tensor)
                pbar.update(50)  # Update progress after model inference
            
            vocals = sources[0, 3].cpu().numpy()  # Vocals (4th stem for htdemucs)
            background = sources[0, :3].sum(dim=0).cpu().numpy()  # Sum of drums, bass, other
            pbar.update(20)  # Update progress
            
            del sources  # Free memory
            torch.cuda.empty_cache()  # Clear GPU cache after heavy operation
            pbar.update(5)  # Final progress update
        
        logger.info(f"Vocal separation completed. Extracted {len(vocals)/self.sample_rate:.2f} seconds of vocals")
        return vocals, background
    
    def perform_diarization(self, audio: np.ndarray) -> List[Dict]:
        """Perform speaker diarization on the audio."""
        logger.info("Performing speaker diarization...")
        duration = len(audio) / self.sample_rate
        logger.info(f"Diarizing {duration:.2f} seconds of audio (this may take a while)...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as temp_file:
            sf.write(temp_file.name, audio, self.sample_rate)
            
            # Create a spinner for diarization (which doesn't have clear progress steps)
            with tqdm(total=100, desc="Speaker diarization", unit="%") as pbar:
                pbar.update(10)  # Update after writing temp file
                
                logger.info("Running speaker diarization model...")
                diarization = self.models["diarization"](temp_file.name)
                pbar.update(80)  # Update after model run
                
                segments = [{"speaker": speaker, "start": turn.start, "end": turn.end} 
                            for turn, _, speaker in diarization.itertracks(yield_label=True)]
                pbar.update(10)  # Final progress update
        
        unique_speakers = set(s["speaker"] for s in segments)
        logger.info(f"Detected {len(unique_speakers)} unique speakers: {', '.join(unique_speakers)}")
        logger.info(f"Found {len(segments)} total speaking segments")
        
        # Log speaking duration per speaker
        speaker_durations = {}
        for seg in segments:
            duration = seg["end"] - seg["start"]
            speaker = seg["speaker"]
            speaker_durations[speaker] = speaker_durations.get(speaker, 0) + duration
        
        for speaker, duration in speaker_durations.items():
            logger.info(f"Speaker {speaker}: {duration:.2f} seconds ({(duration/sum(speaker_durations.values()))*100:.1f}%)")
        
        return segments
    
    def identify_hosts(self, audio: np.ndarray, speaker_segments: List[Dict], 
                       host_samples: Optional[List[np.ndarray]] = None, 
                       host_timestamps: Optional[List[List[Tuple[float, float]]]] = None) -> List[str]:
        """Identify hosts using provided samples or timestamps with confidence check."""
        logger.info("Identifying hosts...")
        hosts = []
        
        if host_samples:
            logger.info(f"Using {len(host_samples)} host sample(s) for identification")
            with tqdm(total=len(host_samples), desc="Processing host samples") as pbar:
                speaker_model = self.models["speaker_embedding"]
                host_embeddings = []
                
                for i, sample in enumerate(host_samples):
                    logger.info(f"Processing host sample {i+1}/{len(host_samples)}: {len(sample)/self.sample_rate:.2f} seconds")
                    embedding = speaker_model.encode_batch(torch.tensor(sample).to(self.device)).squeeze()
                    host_embeddings.append(embedding)
                    pbar.update(1)
                
                logger.info("Computing speaker embeddings for all detected speakers")
                speaker_embeddings = {}
                for speaker in tqdm(set(s["speaker"] for s in speaker_segments), desc="Computing speaker embeddings"):
                    seg_audio = np.concatenate([audio[int(s["start"] * self.sample_rate):int(s["end"] * self.sample_rate)] 
                                              for s in speaker_segments if s["speaker"] == speaker])
                    embedding = speaker_model.encode_batch(torch.tensor(seg_audio).to(self.device)).squeeze()
                    speaker_embeddings[speaker] = embedding
                
                import torch.nn.functional as F
                for i, host_emb in enumerate(host_embeddings):
                    similarities = {spk: F.cosine_similarity(host_emb, emb, dim=0).item() 
                                  for spk, emb in speaker_embeddings.items()}
                    host = max(similarities, key=similarities.get)
                    hosts.append(host)
                    logger.info(f"Host sample {i+1} identified as speaker {host} (similarity: {similarities[host]:.4f})")
                    # Log all similarities for debugging
                    for spk, sim in sorted(similarities.items(), key=lambda x: x[1], reverse=True):
                        logger.info(f"  - Speaker {spk}: similarity {sim:.4f}")
        
        elif host_timestamps:
            logger.info(f"Using {len(host_timestamps)} timestamp file(s) for host identification")
            for i, timestamps in enumerate(host_timestamps):
                logger.info(f"Processing timestamp file {i+1}/{len(host_timestamps)}: {len(timestamps)} segments")
                speaker_counts = {}
                total_duration = 0
                
                for start, end in timestamps:
                    duration = end - start
                    total_duration += duration
                    logger.info(f"Analyzing timestamp segment {start:.2f}s - {end:.2f}s (duration: {duration:.2f}s)")
                    
                    for seg in speaker_segments:
                        if seg["end"] > start and seg["start"] < end:
                            overlap = min(seg["end"], end) - max(seg["start"], start)
                            speaker_counts[seg["speaker"]] = speaker_counts.get(seg["speaker"], 0) + overlap
                
                if speaker_counts:
                    total_time = sum(speaker_counts.values())
                    host = max(speaker_counts, key=speaker_counts.get)
                    confidence = speaker_counts[host] / total_time if total_time > 0 else 0
                    
                    # Log all speaker proportions in this timestamp file
                    logger.info(f"Speaker proportions in timestamp file {i+1}:")
                    for spk, duration in sorted(speaker_counts.items(), key=lambda x: x[1], reverse=True):
                        proportion = duration / total_time if total_time > 0 else 0
                        logger.info(f"  - Speaker {spk}: {duration:.2f}s ({proportion:.2%})")
                    
                    if confidence < 0.7:
                        logger.warning(f"Host {host} identified with low confidence ({confidence:.2%})")
                    hosts.append(host)
                    logger.info(f"Host identified via timestamps: {host} (confidence: {confidence:.2%})")
                else:
                    logger.warning(f"No speakers found in timestamp file {i+1}")
        
        else:
            raise ValueError("Either host_samples or host_timestamps must be provided")
        
        # Remove duplicates
        hosts = list(set(hosts))
        logger.info(f"Final identified hosts: {', '.join(hosts)}")
        return hosts
    
    def transcribe_audio(self, audio: np.ndarray, speaker_segments: List[Dict], cache_path: str) -> List[Dict]:
        """Transcribe audio segments with caching."""
        if os.path.exists(cache_path):
            logger.info(f"Loading cached transcription from {cache_path}")
            with open(cache_path, 'r') as f:
                return json.load(f)
        
        logger.info(f"Starting transcription for {len(speaker_segments)} segments...")
        # TODO: Explore batching inputs to Whisper for better performance if processing large numbers of segments
        whisper_processor = self.models["whisper_processor"]
        whisper_model = self.models["whisper_model"]
        transcribed_segments = []
        
        # Group segments by speaker for better log visibility
        speaker_groups = {}
        for seg in speaker_segments:
            spk = seg["speaker"]
            if spk not in speaker_groups:
                speaker_groups[spk] = []
            speaker_groups[spk].append(seg)
        
        logger.info(f"Transcribing segments for {len(speaker_groups)} speakers")
        
        total_segments = len(speaker_segments)
        total_duration = sum(seg["end"] - seg["start"] for seg in speaker_segments)
        logger.info(f"Total transcription: {total_segments} segments, {total_duration:.2f} seconds")
        
        for i, seg in enumerate(tqdm(speaker_segments, desc="Transcribing segments")):
            start_sample = int(seg["start"] * self.sample_rate)
            end_sample = min(int(seg["end"] * self.sample_rate), len(audio))
            
            if start_sample >= end_sample:
                logger.warning(f"Skipping empty segment: {seg['start']:.2f}s - {seg['end']:.2f}s")
                continue
                
            segment_audio = audio[start_sample:end_sample]
            segment_duration = (end_sample - start_sample) / self.sample_rate
            
            # Log every 10% of segments or at least every 50 segments
            log_interval = max(1, min(total_segments // 10, 50))
            if i % log_interval == 0:
                logger.info(f"Transcribing segment {i+1}/{total_segments}: Speaker {seg['speaker']} ({segment_duration:.2f}s)")
            
            inputs = whisper_processor(segment_audio, sampling_rate=self.sample_rate, return_tensors="pt").input_features.to(self.device)
            if self.use_fp16:
                inputs = inputs.half()
            
            with torch.no_grad():
                predicted_ids = whisper_model.generate(inputs)
            
            text = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()
            
            # Only log very short or very long segments for debugging
            if segment_duration < 0.5 or segment_duration > 10:
                logger.info(f"Segment {i+1}: {segment_duration:.2f}s, {len(text)} chars: '{text[:50]}...'")
                
            transcribed_segments.append({
                "speaker": seg["speaker"], 
                "start": seg["start"], 
                "end": seg["end"], 
                "text": text
            })
        
        # Log summary statistics
        word_counts = {}
        for seg in transcribed_segments:
            spk = seg["speaker"]
            words = len(seg["text"].split())
            word_counts[spk] = word_counts.get(spk, 0) + words
        
        logger.info("Transcription statistics per speaker:")
        for spk, words in word_counts.items():
            logger.info(f"Speaker {spk}: {words} words")
        
        with open(cache_path, 'w') as f:
            json.dump(transcribed_segments, f)
        logger.info(f"Transcription completed, cached to {cache_path}")
        return transcribed_segments
    
    def extract_voice_embedding(self, reference_audio: np.ndarray) -> np.ndarray:
        """Extract speaker embedding from reference audio.
        
        Warning: This method is hypothetical and assumes MetaVoice-1B functionality.
        When debugging, verify the actual method name (e.g., encode_speaker_embedding) and input format against MetaVoice-1B API docs.
        The processor and model outputs may differ from this implementation.
        """
        logger.info("Extracting voice embedding from reference sample...")
        logger.info(f"Reference audio duration: {len(reference_audio)/self.sample_rate:.2f} seconds")
        
        processor = self.models["metavoice_processor"]
        model = self.models["metavoice_model"]
        
        with tqdm(total=100, desc="Extracting voice embedding", unit="%") as pbar:
            pbar.update(10)
            inputs = processor(audio=reference_audio, sampling_rate=self.sample_rate, return_tensors="pt").to(self.device)
            pbar.update(30)
            
            if self.use_fp16:
                inputs = {k: v.half() for k, v in inputs.items()}  # Assuming inputs is a dict; check actual structure
            pbar.update(10)
            
            with torch.no_grad():
                logger.info("Running voice embedding extraction...")
                embedding = model.encode_speaker_embedding(inputs.input_values).cpu().numpy()
                pbar.update(50)
        
        logger.info("Voice embedding extraction completed")
        return embedding
    
    def synthesize_voice(self, text: str, voice_embedding: np.ndarray) -> np.ndarray:
        """Synthesize speech with the target voice.
        
        Warning: This method is hypothetical and assumes MetaVoice-1B capabilities.
        Check MetaVoice-1B documentation for the correct synthesis method (e.g., generate), parameters, and tensor shapes.
        If errors occur, inspect the processor output and model.generate signature.
        """
        # Only log beginning of long texts
        log_text = text[:50] + "..." if len(text) > 50 else text
        logger.info(f"Synthesizing text ({len(text)} chars): '{log_text}'")
        
        processor = self.models["metavoice_processor"]
        model = self.models["metavoice_model"]
        
        inputs = processor(text=text, return_tensors="pt").to(self.device)
        if self.use_fp16:
            inputs = {k: v.half() for k, v in inputs.items()}  # Assuming dict; verify with actual API
        
        embedding_tensor = torch.tensor(voice_embedding).to(self.device)
        
        with torch.no_grad():
            audio = model.generate(**inputs, speaker_embeddings=embedding_tensor).cpu().numpy().squeeze()
        
        logger.info(f"Synthesized {len(audio)/self.sample_rate:.2f} seconds of audio")
        return audio
    
    def process_podcast(self, podcast_path: str, reference_voice_path: str, output_path: str,
                        host_samples_paths: Optional[List[str]] = None, 
                        host_timestamps_files: Optional[List[str]] = None):
        """Process the podcast and replace host voices."""
        logger.info("=" * 80)
        logger.info("STARTING PODCAST VOICE REPLACEMENT")
        logger.info("=" * 80)
        logger.info(f"Podcast: {podcast_path}")
        logger.info(f"Reference voice: {reference_voice_path}")
        logger.info(f"Output: {output_path}")
        if host_samples_paths:
            logger.info(f"Host samples: {', '.join(host_samples_paths)}")
        if host_timestamps_files:
            logger.info(f"Host timestamps: {', '.join(host_timestamps_files)}")
        logger.info("-" * 80)
        
        # Load and resample all audio to 16 kHz
        logger.info("\n[STEP 1/8] Loading audio files")
        podcast_audio, _ = self.load_audio(podcast_path)
        reference_audio, _ = self.load_audio(reference_voice_path)
        host_samples = [self.load_audio(path)[0] for path in host_samples_paths] if host_samples_paths else None
        host_timestamps = None
        if host_timestamps_files:
            host_timestamps = []
            for file in host_timestamps_files:
                logger.info(f"Loading host timestamps from {file}")
                with open(file, 'r') as f:
                    data = json.load(f)
                    timestamps = [(d["start"], d["end"]) if isinstance(d, dict) else (d[0], d[1]) for d in data]
                    logger.info(f"Loaded {len(timestamps)} timestamp ranges")
                    host_timestamps.append(timestamps)
        
        # Separate vocals and background
        logger.info("\n[STEP 2/8] Separating vocals from background")
        vocals, background = self.separate_vocals(podcast_audio)
        
        # Diarize and transcribe
        logger.info("\n[STEP 3/8] Performing speaker diarization")
        speaker_segments = self.perform_diarization(podcast_audio)
        
        logger.info("\n[STEP 4/8] Transcribing audio segments")
        cache_path = podcast_path + ".transcription.json"
        transcribed_segments = self.transcribe_audio(podcast_audio, speaker_segments, cache_path)
        
        # Identify hosts
        logger.info("\n[STEP 5/8] Identifying podcast hosts")
        hosts = self.identify_hosts(podcast_audio, speaker_segments, host_samples, host_timestamps)
        
        # Extract reference voice embedding using hypothetical MetaVoice method
        logger.info("\n[STEP 6/8] Extracting reference voice characteristics")
        voice_embedding = self.extract_voice_embedding(reference_audio)
        
        # Synthesize host segments
        logger.info("\n[STEP 7/8] Synthesizing new host voice segments")
        # TODO: If MetaVoice supports batching, synthesizing multiple texts at once could improve performance
        synthesized_audio = np.zeros_like(podcast_audio)
        
        # Filter segments for hosts only and count non-empty text segments
        host_segments = [seg for seg in transcribed_segments if seg["speaker"] in hosts and seg["text"].strip()]
        logger.info(f"Synthesizing {len(host_segments)} host segments ({len(host_segments)/len(transcribed_segments):.1%} of total)")
        
        total_host_duration = sum(seg["end"] - seg["start"] for seg in host_segments)
        logger.info(f"Total host speaking time: {total_host_duration:.2f} seconds")
        
        for seg in tqdm(host_segments, desc="Synthesizing host segments"):
            duration = seg["end"] - seg["start"]
            text = seg["text"].strip()
            
            if not text:
                logger.info(f"Skipping empty text segment: {seg['start']:.2f}s - {seg['end']:.2f}s")
                continue
                
            # Only log for longer segments or very short ones (might indicate issues)
            if duration > 5 or duration < 0.5:
                logger.info(f"Synthesizing {duration:.2f}s segment: '{text[:50]}...'")
                
            audio = self.synthesize_voice(text, voice_embedding)
            target_len = int((seg["end"] - seg["start"]) * self.sample_rate)
            
            # Log if large time stretching is needed (might indicate issues)
            stretch_ratio = len(audio) / target_len if target_len > 0 else 0
            if abs(stretch_ratio - 1) > 0.3:  # If more than 30% stretching needed
                logger.info(f"Time stretching needed: {stretch_ratio:.2f}x (synthesized: {len(audio)/self.sample_rate:.2f}s, target: {target_len/self.sample_rate:.2f}s)")
                
            if len(audio) != target_len:
                audio = librosa.effects.time_stretch(audio, rate=len(audio) / target_len)
                
            if len(audio) > target_len:
                audio = audio[:target_len]
            elif len(audio) < target_len:
                audio = np.pad(audio, (0, target_len - len(audio)))
                
            start = int(seg["start"] * self.sample_rate)
            end = min(start + target_len, len(synthesized_audio))
            synthesized_audio[start:end] = audio[:end - start]
        
        # Mix audio with fade transitions
        logger.info("\n[STEP 8/8] Mixing final audio with background")
        host_mask = np.zeros_like(podcast_audio)
        fade_samples = int(0.05 * self.sample_rate)  # 50ms fade
        
        logger.info(f"Applying crossfade transitions ({fade_samples/self.sample_rate*1000:.0f}ms)")
        for seg in tqdm(transcribed_segments, desc="Creating audio mask"):
            if seg["speaker"] in hosts:
                start = int(seg["start"] * self.sample_rate)
                end = min(int(seg["end"] * self.sample_rate), len(host_mask))
                
                if start + fade_samples < end - fade_samples:
                    host_mask[start:start + fade_samples] = np.linspace(0, 1, fade_samples)
                    host_mask[start + fade_samples:end - fade_samples] = 1
                    host_mask[end - fade_samples:end] = np.linspace(1, 0, fade_samples)
        
        mixed_vocals = vocals * (1 - host_mask) + synthesized_audio * host_mask
        final_audio = mixed_vocals + background
        
        # Normalize audio to prevent clipping
        logger.info("Normalizing final audio")
        max_amplitude = np.max(np.abs(final_audio))
        if max_amplitude > 0:
            normalization_factor = 0.9 / max_amplitude
            logger.info(f"Applying normalization factor: {normalization_factor:.4f}")
            final_audio = final_audio * normalization_factor
        
        # Save output at 16 kHz
        logger.info(f"Saving output to {output_path}")
        temp_wav = output_path + ".temp.wav"
        sf.write(temp_wav, final_audio, self.sample_rate)
        
        output_format = 'mp3' if output_path.lower().endswith('.mp3') else 'wav'
        logger.info(f"Converting to final {output_format.upper()} format")
        
        audio_seg = AudioSegment.from_wav(temp_wav)
        audio_seg.export(output_path, format=output_format)
        os.remove(temp_wav)
        
        # Log final statistics
        final_duration = len(final_audio) / self.sample_rate
        logger.info(f"Output saved to {output_path} ({final_duration:.2f} seconds)")
        logger.info(f"Original host speaking time: {total_host_duration:.2f}s ({(total_host_duration/final_duration)*100:.1f}% of podcast)")
        logger.info("=" * 80)
        logger.info("PODCAST VOICE REPLACEMENT COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
        torch.cuda.empty_cache()  # Clear GPU memory after processing

def main():
    parser = argparse.ArgumentParser(description="Podcast Voice Replacement Tool")
    parser.add_argument("podcast", help="Path to podcast audio file")
    parser.add_argument("reference_voice", help="Path to reference voice sample")
    parser.add_argument("output", help="Path to save output audio")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--host-samples", nargs="+", help="Paths to host voice sample files")
    group.add_argument("--host-timestamps", nargs="+", help="JSON files with host timestamp ranges")
    parser.add_argument("--hf-token", required=True, help="Hugging Face token for model access")
    parser.add_argument("--fp16", action="store_true", default=False, help="Use FP16 precision")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Processing device")
    parser.add_argument("--verbose", action="store_true", help="Enable more detailed logging")
    
    args = parser.parse_args()
    
    # Set more detailed logging if verbose flag is used
    if args.verbose:
        logging.getLogger("podcast_voice_replacement").setLevel(logging.DEBUG)
        logger.info("Verbose logging enabled")
    
    start_time = time.time()
    
    try:
        logger.info(f"Initializing AudioProcessor with {args.device} device, FP16={args.fp16}")
        processor = AudioProcessor(use_fp16=args.fp16, device=args.device, hf_token=args.hf_token)
        
        logger.info("Loading required models...")
        processor.load_models()
        
        logger.info("Starting podcast processing pipeline...")
        processor.process_podcast(
            podcast_path=args.podcast,
            reference_voice_path=args.reference_voice,
            output_path=args.output,
            host_samples_paths=args.host_samples,
            host_timestamps_files=args.host_timestamps
        )
        
        total_time = time.time() - start_time
        minutes, seconds = divmod(total_time, 60)
        logger.info(f"Processing completed in {int(minutes)}m {seconds:.2f}s")
        logger.info(f"Output saved to: {args.output}")
        
    except Exception as e:
        logger.error(f"An error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()
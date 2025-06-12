"""Generic Audio Post-Processor Stage for audio enhancement and cleanup.

This stage provides common audio processing techniques to improve quality:
- High-pass filtering: Remove rumble and low-frequency noise
- Noise gating: Reduce background noise during quiet passages  
- Dynamic compression: Even out volume levels
- Spectral enhancement: Improve clarity and presence
- Limiting: Prevent clipping and control peaks

WHEN TO USE:
- After source separation to clean up ML artifacts
- To enhance vocals for transcription or synthesis
- To prepare audio for broadcast/streaming
- To improve poor quality recordings

REFERENCES:
- Audio Processing Guide: https://www.soundonsound.com/techniques
- Digital Audio Effects: https://www.dafx.de/
- Audio Engineering Basics: https://www.prosoundnetwork.com/audio
- Python Audio Processing: https://librosa.org/doc/latest/
"""

import numpy as np
from typing import Dict, Any, Tuple, List, Optional
from dataclasses import dataclass

from ..stage_interface import (
    EnhancedPipelineStage, StageMetadata, StageInput, StageOutput,
    DataType
)


@dataclass
class ProcessingPreset:
    """Pre-configured settings for common use cases."""
    name: str
    description: str
    high_pass_cutoff: Optional[float]  # Hz
    noise_gate_threshold: float        # dB
    spectral_enhancement: bool
    dynamic_compression: bool
    compression_ratio: float           # X:1
    limiter_threshold: float          # dB


class AudioPostProcessor(EnhancedPipelineStage):
    """Generic audio post-processing stage for enhancement and cleanup."""

    # Quick presets for common scenarios
    PRESETS = {
        "clean": ProcessingPreset("clean", "Light processing for good audio", 
                                 40.0, -70.0, False, False, 2.0, -1.0),
        "enhance": ProcessingPreset("enhance", "Balanced enhancement [DEFAULT]", 
                                   80.0, -60.0, True, True, 3.0, -0.5),
        "aggressive": ProcessingPreset("aggressive", "Heavy processing for problem audio", 
                                      120.0, -50.0, True, True, 4.0, -0.1),
        "vocal": ProcessingPreset("vocal", "Optimized for speech/vocals", 
                                 85.0, -55.0, True, True, 3.5, -0.3),
        "music": ProcessingPreset("music", "Preserve musical character", 
                                 30.0, -65.0, False, True, 2.5, -0.5)
    }

    def __init__(self, stage_id: str = None, config: Dict[str, Any] = None):
        super().__init__(stage_id, config)
        
        # Preset selection (clean, enhance, aggressive, vocal, music)
        self.preset = self.config.get("preset", "enhance")
        preset_config = self.PRESETS.get(self.preset, self.PRESETS["enhance"])
        
        # Input/output configuration
        self.input_sources = self.config.get("input_sources", ["auto"])  # ["vocals", "music"] or ["auto"]
        self.output_suffix = self.config.get("output_suffix", "_enhanced")
        
        # High-pass filter - removes low frequencies (rumble, handling noise)
        # Reference: https://en.wikipedia.org/wiki/High-pass_filter
        self.high_pass_enabled = self.config.get("high_pass_enabled", preset_config.high_pass_cutoff is not None)
        self.high_pass_cutoff = self.config.get("high_pass_cutoff", preset_config.high_pass_cutoff or 80.0)  # Hz
        
        # Noise gate - reduces background noise in quiet sections
        # Reference: https://en.wikipedia.org/wiki/Noise_gate
        self.noise_gate_enabled = self.config.get("noise_gate_enabled", True)
        self.noise_gate_threshold = self.config.get("noise_gate_threshold", preset_config.noise_gate_threshold)  # dB
        
        # Spectral enhancement - improves clarity with pre-emphasis
        # Reference: https://en.wikipedia.org/wiki/Pre-emphasis
        self.spectral_enhancement = self.config.get("spectral_enhancement", preset_config.spectral_enhancement)
        self.preemphasis_coeff = self.config.get("preemphasis_coeff", 0.97)  # 0.9-0.99
        
        # Dynamic compression - evens out volume levels
        # Reference: https://en.wikipedia.org/wiki/Dynamic_range_compression
        self.dynamic_compression = self.config.get("dynamic_compression", preset_config.dynamic_compression)
        self.compression_threshold = self.config.get("compression_threshold", -20.0)  # dB
        self.compression_ratio = self.config.get("compression_ratio", preset_config.compression_ratio)  # X:1
        
        # Limiter - prevents clipping and controls peaks
        # Reference: https://en.wikipedia.org/wiki/Limiter
        self.limiter_enabled = self.config.get("limiter_enabled", True)
        self.limiter_threshold = self.config.get("limiter_threshold", preset_config.limiter_threshold)  # dB
        
        self.sample_rate = self.config.get("sample_rate", 44100)

    def get_metadata(self) -> StageMetadata:
        """Define stage inputs, outputs, and configuration options."""
        
        preset_info = self.PRESETS.get(self.preset, self.PRESETS["enhance"])
        
        # Build feature list
        features = []
        if self.high_pass_enabled:
            features.append(f"High-pass: {self.high_pass_cutoff}Hz")
        if self.noise_gate_enabled:
            features.append(f"Noise gate: {self.noise_gate_threshold}dB")
        if self.spectral_enhancement:
            features.append("Spectral enhancement")
        if self.dynamic_compression:
            features.append(f"Compression: {self.compression_ratio}:1")
        if self.limiter_enabled:
            features.append(f"Limiter: {self.limiter_threshold}dB")
        
        config_docs = f"""
Configuration Options:

**Presets** (preset: "{self.preset}"):
  • "clean": Light processing for good quality audio
  • "enhance": Balanced processing [DEFAULT]
  • "aggressive": Heavy processing for problem audio  
  • "vocal": Optimized for speech/vocals
  • "music": Preserve musical character

**Processing Features**:
  • high_pass_enabled ({self.high_pass_enabled}): Remove low frequencies
  • noise_gate_enabled ({self.noise_gate_enabled}): Reduce background noise
  • spectral_enhancement ({self.spectral_enhancement}): Improve clarity
  • dynamic_compression ({self.dynamic_compression}): Even out levels
  • limiter_enabled ({self.limiter_enabled}): Prevent clipping

**Fine-tuning**:
  • high_pass_cutoff ({self.high_pass_cutoff} Hz): Frequency cutoff
  • noise_gate_threshold ({self.noise_gate_threshold} dB): Gate threshold
  • compression_ratio ({self.compression_ratio}:1): Compression amount
  • limiter_threshold ({self.limiter_threshold} dB): Maximum level

**Learn More**:
  • Audio Processing: https://www.soundonsound.com/techniques
  • Digital Effects: https://www.dafx.de/
  • Python Audio: https://librosa.org/doc/latest/
"""
        
        return StageMetadata(
            name="Audio Post-Processor",
            description=f"Generic audio enhancement using {preset_info.description.lower()}" + config_docs,
            category="post_processing",
            model_name=f"Processing preset: {self.preset}",
            performance_notes=f"Active features: {', '.join(features) if features else 'None'}",
            inputs=[
                StageInput("audio_input", DataType.AUDIO_WITH_SR, False, "Single audio source"),
                StageInput("separated_audio", DataType.SEPARATED_AUDIO, False, "Multiple audio sources"),
                StageInput("vocals", DataType.AUDIO_MONO, False, "Vocal audio"),
                StageInput("music", DataType.AUDIO_MONO, False, "Music audio")
            ],
            outputs=[
                StageOutput("processed_audio", DataType.SEPARATED_AUDIO, "Enhanced audio sources"),
                StageOutput("quality_metrics", DataType.TRANSCRIPTION, "Processing statistics")
            ]
        )

    async def process(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process audio through the enhancement pipeline."""
        await self.validate_inputs(inputs)
        
        # Extract audio sources
        audio_sources, sample_rate = self._extract_audio_sources(inputs)
        if not audio_sources:
            raise ValueError("No audio sources found in inputs")
        
        processed_sources = {}
        metrics = {
            "preset": self.preset,
            "sources_processed": [],
            "processing_stats": {}
        }
        
        # Process each audio source
        for source_name, audio_data in audio_sources.items():
            if not self._should_process_source(source_name):
                processed_sources[source_name] = audio_data
                continue
            
            try:
                # Apply processing chain
                processed_audio, source_metrics = self._process_audio_chain(audio_data, sample_rate)
                
                # Store results
                output_name = f"{source_name}{self.output_suffix}"
                processed_sources[output_name] = processed_audio
                metrics["sources_processed"].append(source_name)
                metrics["processing_stats"][source_name] = source_metrics
                
            except Exception as e:
                # Fall back to original on error
                processed_sources[source_name] = audio_data
                metrics["processing_stats"][source_name] = {"error": str(e)}
        
        return {
            "processed_audio": processed_sources,
            "quality_metrics": metrics
        }

    def _extract_audio_sources(self, inputs: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], int]:
        """Extract audio from various input formats."""
        audio_sources = {}
        sample_rate = self.sample_rate
        
        # Check separated_audio dict
        if "separated_audio" in inputs and isinstance(inputs["separated_audio"], dict):
            for key, audio in inputs["separated_audio"].items():
                if isinstance(audio, np.ndarray) and audio.size > 0:
                    audio_sources[key] = audio
        
        # Check individual sources
        for key in ["vocals", "music", "drums", "bass", "other", "audio"]:
            if key in inputs and isinstance(inputs[key], np.ndarray) and inputs[key].size > 0:
                audio_sources[key] = inputs[key]
        
        # Check audio_input tuple
        if "audio_input" in inputs:
            audio_input = inputs["audio_input"]
            if isinstance(audio_input, tuple) and len(audio_input) == 2:
                audio, sr = audio_input
                if isinstance(audio, np.ndarray) and audio.size > 0:
                    audio_sources["audio"] = audio
                    sample_rate = sr
        
        return audio_sources, sample_rate

    def _should_process_source(self, source_name: str) -> bool:
        """Check if source should be processed."""
        return "auto" in self.input_sources or source_name in self.input_sources

    def _process_audio_chain(self, audio: np.ndarray, sample_rate: int) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply processing chain: high-pass → spectral → compression → gate → limiter."""
        processed = audio.copy()
        
        metrics = {
            "original_peak_db": 20 * np.log10(np.max(np.abs(audio))) if np.max(np.abs(audio)) > 0 else -np.inf,
            "steps_applied": []
        }
        
        # 1. High-pass filter
        if self.high_pass_enabled:
            processed = self._apply_high_pass_filter(processed, sample_rate)
            metrics["steps_applied"].append("high_pass")
        
        # 2. Spectral enhancement
        if self.spectral_enhancement:
            processed = self._apply_spectral_enhancement(processed)
            metrics["steps_applied"].append("spectral_enhancement")
        
        # 3. Compression
        if self.dynamic_compression:
            processed = self._apply_compression(processed)
            metrics["steps_applied"].append("compression")
        
        # 4. Noise gate
        if self.noise_gate_enabled:
            processed = self._apply_noise_gate(processed, sample_rate)
            metrics["steps_applied"].append("noise_gate")
        
        # 5. Limiter
        if self.limiter_enabled:
            processed = self._apply_limiter(processed)
            metrics["steps_applied"].append("limiter")
        
        metrics["final_peak_db"] = 20 * np.log10(np.max(np.abs(processed))) if np.max(np.abs(processed)) > 0 else -np.inf
        metrics["peak_change_db"] = metrics["final_peak_db"] - metrics["original_peak_db"]
        
        return processed, metrics

    def _apply_high_pass_filter(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Remove low frequencies using Butterworth filter."""
        try:
            import scipy.signal as signal
            
            nyquist = sample_rate / 2
            normalized_cutoff = self.high_pass_cutoff / nyquist
            
            if normalized_cutoff >= 1.0 or normalized_cutoff <= 0:
                return audio
            
            b, a = signal.butter(2, normalized_cutoff, btype='high')
            
            if audio.ndim == 1:
                return signal.filtfilt(b, a, audio)
            else:
                filtered = np.zeros_like(audio)
                for i in range(audio.shape[0]):
                    filtered[i] = signal.filtfilt(b, a, audio[i])
                return filtered
                
        except ImportError:
            return audio  # Skip if scipy not available

    def _apply_spectral_enhancement(self, audio: np.ndarray) -> np.ndarray:
        """Apply pre-emphasis for improved clarity."""
        if self.preemphasis_coeff <= 0 or self.preemphasis_coeff >= 1:
            return audio
        
        enhanced = audio.copy()
        
        if audio.ndim == 1:
            enhanced = np.append(enhanced[0], enhanced[1:] - self.preemphasis_coeff * enhanced[:-1])
        else:
            for i in range(enhanced.shape[0]):
                enhanced[i] = np.append(
                    enhanced[i, 0], 
                    enhanced[i, 1:] - self.preemphasis_coeff * enhanced[i, :-1]
                )
        
        return enhanced

    def _apply_compression(self, audio: np.ndarray) -> np.ndarray:
        """Simple peak-based compression."""
        threshold_linear = 10 ** (self.compression_threshold / 20)
        peak = np.max(np.abs(audio))
        
        if peak > threshold_linear:
            excess_db = 20 * np.log10(peak / threshold_linear)
            compressed_excess_db = excess_db / self.compression_ratio
            target_peak = threshold_linear * (10 ** (compressed_excess_db / 20))
            gain = target_peak / peak
            return audio * gain
        
        return audio

    def _apply_noise_gate(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
        """Reduce background noise using RMS-based gating."""
        threshold_linear = 10 ** (self.noise_gate_threshold / 20)
        window_size = int(sample_rate * 0.01)  # 10ms windows
        hop_size = window_size // 2
        
        gated = audio.copy()
        
        if audio.ndim == 1:
            for i in range(0, len(audio) - window_size, hop_size):
                window = audio[i:i + window_size]
                rms = np.sqrt(np.mean(window ** 2))
                if rms < threshold_linear:
                    attenuation = max(0.1, rms / threshold_linear)
                    gated[i:i + window_size] *= attenuation
        else:
            for channel in range(audio.shape[0]):
                for i in range(0, audio.shape[1] - window_size, hop_size):
                    window = audio[channel, i:i + window_size]
                    rms = np.sqrt(np.mean(window ** 2))
                    if rms < threshold_linear:
                        attenuation = max(0.1, rms / threshold_linear)
                        gated[channel, i:i + window_size] *= attenuation
        
        return gated

    def _apply_limiter(self, audio: np.ndarray) -> np.ndarray:
        """Prevent clipping with brick-wall limiter."""
        threshold_linear = 10 ** (self.limiter_threshold / 20)
        return np.clip(audio, -threshold_linear, threshold_linear)

    async def verify_outputs(self, outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Override base verification to handle AudioPostProcessor-specific output format."""
        messages = []
        success = True
        
        try:
            # Basic structure verification
            if not isinstance(outputs, dict):
                messages.append(f"❌ Expected dict output, got {type(outputs)}")
                return False, messages
            
            # Check for required outputs
            if "processed_audio" not in outputs:
                messages.append("❌ Missing 'processed_audio' output")
                success = False
            
            if "quality_metrics" not in outputs:
                messages.append("❌ Missing 'quality_metrics' output")
                success = False
            
            # Verify processed_audio structure
            if "processed_audio" in outputs:
                processed = outputs["processed_audio"]
                if isinstance(processed, dict):
                    messages.append(f"✅ Processed audio contains {len(processed)} sources: {list(processed.keys())}")
                    
                    # Check each audio output
                    for name, audio in processed.items():
                        if isinstance(audio, np.ndarray) and audio.size > 0:
                            if np.isnan(audio).any():
                                messages.append(f"❌ NaN values in {name}")
                                success = False
                            elif np.max(np.abs(audio)) > 1.0:
                                messages.append(f"⚠️ Clipping detected in {name}")
                            elif np.all(audio == 0):
                                messages.append(f"⚠️ Silent audio in {name}")
                                success = False
                            else:
                                peak_db = 20 * np.log10(np.max(np.abs(audio))) if np.max(np.abs(audio)) > 0 else -np.inf
                                messages.append(f"✅ {name}: {len(audio):,} samples, peak {peak_db:.1f}dB")
                        else:
                            messages.append(f"❌ Invalid audio data in {name}")
                            success = False
                else:
                    messages.append(f"❌ processed_audio should be dict, got {type(processed)}")
                    success = False
            
            # Verify quality_metrics structure
            if "quality_metrics" in outputs:
                metrics = outputs["quality_metrics"]
                if isinstance(metrics, dict):
                    preset = metrics.get("preset", "unknown")
                    sources = metrics.get("sources_processed", [])
                    messages.append(f"✅ Quality metrics: {preset} preset, {len(sources)} sources processed")
                else:
                    messages.append(f"❌ quality_metrics should be dict, got {type(metrics)}")
                    success = False
            
            # Run stage-specific verification
            stage_success, stage_messages = await self.verify_stage_specific(outputs)
            messages.extend(stage_messages)
            if not stage_success:
                success = False
            
            if success:
                messages.append("✅ All AudioPostProcessor outputs verified successfully")
            
            return success, messages
            
        except Exception as e:
            messages.append(f"❌ Verification error: {e}")
            return False, messages

    async def verify_stage_specific(self, outputs: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Verify post-processing results."""
        messages = []
        success = True
        
        messages.append(f"🎛️ Audio Post-Processor: {self.preset} preset")
        
        if "quality_metrics" in outputs:
            metrics = outputs["quality_metrics"]
            sources = metrics.get("sources_processed", [])
            messages.append(f"📊 Processed {len(sources)} sources: {sources}")
            
            for source, stats in metrics.get("processing_stats", {}).items():
                if "error" in stats:
                    messages.append(f"❌ Error in {source}: {stats['error']}")
                    success = False
                else:
                    peak_change = stats.get("peak_change_db", 0)
                    steps = ", ".join(stats.get("steps_applied", []))
                    messages.append(f"✅ {source}: {peak_change:+.1f}dB peak change, steps: {steps}")
        
        if "processed_audio" in outputs:
            processed = outputs["processed_audio"]
            if isinstance(processed, dict):
                for name, audio in processed.items():
                    if isinstance(audio, np.ndarray):
                        if np.isnan(audio).any():
                            messages.append(f"❌ NaN values in {name}")
                            success = False
                        elif np.max(np.abs(audio)) > 1.0:
                            messages.append(f"⚠️ Clipping in {name}")
                        elif np.all(audio == 0):
                            messages.append(f"⚠️ Silent audio in {name}")
        
        messages.append("✅ Post-processing verification completed")
        return success, messages
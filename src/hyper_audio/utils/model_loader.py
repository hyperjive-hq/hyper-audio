"""Reusable model loading utilities with progress feedback."""

import torch
from typing import Any, Optional, Dict, Callable
from pathlib import Path
from tqdm import tqdm
import time

from .logging_utils import get_logger
from ..config.settings import settings

logger = get_logger("model_loader")


class ModelLoader:
    """Centralized model loading with caching and progress feedback."""
    
    def __init__(self):
        self.device = torch.device(settings.device)
        self.cache_dir = settings.model_cache_dir
        self._model_cache: Dict[str, Any] = {}
        
        logger.info(f"ModelLoader initialized with device: {self.device}")
        logger.info(f"Model cache directory: {self.cache_dir}")
    
    async def load_demucs_model(self, model_name: str = "htdemucs") -> Any:
        """Load Demucs model with progress feedback.
        
        Args:
            model_name: Name of the Demucs model to load
            
        Returns:
            Loaded Demucs model
        """
        cache_key = f"demucs_{model_name}"
        
        if cache_key in self._model_cache:
            logger.info(f"Using cached Demucs model: {model_name}")
            return self._model_cache[cache_key]
        
        try:
            self._show_loading_message("Demucs", model_name, "voice/music separation")
            
            # Import demucs here to avoid startup delays
            from demucs import pretrained
            
            with tqdm(desc="Loading Demucs", unit="step", total=3) as pbar:
                pbar.set_description("🔄 Downloading/Loading Demucs model")
                model = pretrained.get_model(name=model_name)
                pbar.update(1)
                
                pbar.set_description("📦 Moving model to device")
                model = model.to(self.device)
                pbar.update(1)
                
                pbar.set_description("✅ Model ready")
                pbar.update(1)
            
            self._model_cache[cache_key] = model
            self._show_success_message("Demucs", model_name)
            
            return model
            
        except ImportError as e:
            error_msg = (
                f"Failed to import Demucs. Please install with:\n"
                f"  pip install demucs\n"
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load Demucs model '{model_name}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def load_whisper_model(self, model_name: str = "large-v2") -> Any:
        """Load Whisper model with progress feedback.
        
        Args:
            model_name: Name of the Whisper model to load
            
        Returns:
            Loaded Whisper model
        """
        cache_key = f"whisper_{model_name}"
        
        if cache_key in self._model_cache:
            logger.info(f"Using cached Whisper model: {model_name}")
            return self._model_cache[cache_key]
        
        try:
            self._show_loading_message("Whisper", model_name, "speech recognition")
            
            import whisper
            
            with tqdm(desc="Loading Whisper", unit="step", total=3) as pbar:
                pbar.set_description("🔄 Downloading/Loading Whisper model")
                model = whisper.load_model(model_name, device=self.device)
                pbar.update(1)
                
                pbar.set_description("📦 Model configuration")
                # Additional setup if needed
                pbar.update(1)
                
                pbar.set_description("✅ Model ready")
                pbar.update(1)
            
            self._model_cache[cache_key] = model
            self._show_success_message("Whisper", model_name)
            
            return model
            
        except ImportError as e:
            error_msg = (
                f"Failed to import Whisper. Please install with:\n"
                f"  pip install openai-whisper\n"
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load Whisper model '{model_name}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def load_huggingface_model(
        self, 
        model_id: str, 
        model_class: Any,
        tokenizer_class: Optional[Any] = None,
        **model_kwargs
    ) -> Any:
        """Load HuggingFace model with progress feedback.
        
        Args:
            model_id: HuggingFace model identifier
            model_class: Model class to instantiate
            tokenizer_class: Optional tokenizer class
            **model_kwargs: Additional arguments for model loading
            
        Returns:
            Loaded model (and tokenizer if requested)
        """
        cache_key = f"hf_{model_id.replace('/', '_')}"
        
        if cache_key in self._model_cache:
            logger.info(f"Using cached HuggingFace model: {model_id}")
            return self._model_cache[cache_key]
        
        try:
            self._show_loading_message("HuggingFace", model_id, "AI processing")
            
            result = {}
            
            with tqdm(desc="Loading HF Model", unit="step", total=4) as pbar:
                pbar.set_description("🔄 Downloading/Loading model")
                model = model_class.from_pretrained(
                    model_id,
                    cache_dir=self.cache_dir,
                    **model_kwargs
                )
                result["model"] = model
                pbar.update(1)
                
                if tokenizer_class:
                    pbar.set_description("📝 Loading tokenizer")
                    tokenizer = tokenizer_class.from_pretrained(
                        model_id,
                        cache_dir=self.cache_dir
                    )
                    result["tokenizer"] = tokenizer
                    pbar.update(1)
                else:
                    pbar.update(1)
                
                pbar.set_description("📦 Moving to device")
                if hasattr(model, 'to'):
                    model = model.to(self.device)
                    result["model"] = model
                pbar.update(1)
                
                pbar.set_description("✅ Model ready")
                pbar.update(1)
            
            # Return model only if no tokenizer, otherwise return dict
            final_result = result["model"] if len(result) == 1 else result
            self._model_cache[cache_key] = final_result
            self._show_success_message("HuggingFace", model_id)
            
            return final_result
            
        except ImportError as e:
            error_msg = (
                f"Failed to import HuggingFace transformers. Please install with:\n"
                f"  pip install transformers\n"
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load HuggingFace model '{model_id}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    async def load_pyannote_model(self, model_id: str, auth_token: Optional[str] = None) -> Any:
        """Load pyannote.audio model with progress feedback.
        
        Args:
            model_id: pyannote model identifier
            auth_token: HuggingFace auth token if needed
            
        Returns:
            Loaded pyannote model
        """
        cache_key = f"pyannote_{model_id.replace('/', '_')}"
        
        if cache_key in self._model_cache:
            logger.info(f"Using cached pyannote model: {model_id}")
            return self._model_cache[cache_key]
        
        try:
            self._show_loading_message("pyannote.audio", model_id, "speaker diarization")
            
            from pyannote.audio import Pipeline
            
            with tqdm(desc="Loading pyannote", unit="step", total=3) as pbar:
                pbar.set_description("🔄 Downloading/Loading pyannote model")
                pipeline = Pipeline.from_pretrained(
                    model_id,
                    use_auth_token=auth_token or settings.huggingface_token
                )
                pbar.update(1)
                
                pbar.set_description("📦 Moving to device")
                pipeline = pipeline.to(self.device)
                pbar.update(1)
                
                pbar.set_description("✅ Model ready")
                pbar.update(1)
            
            self._model_cache[cache_key] = pipeline
            self._show_success_message("pyannote.audio", model_id)
            
            return pipeline
            
        except ImportError as e:
            error_msg = (
                f"Failed to import pyannote.audio. Please install with:\n"
                f"  pip install pyannote.audio\n"
                f"Error: {e}"
            )
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load pyannote model '{model_id}': {e}"
            logger.error(error_msg)
            raise RuntimeError(error_msg) from e
    
    def _show_loading_message(self, model_type: str, model_name: str, purpose: str):
        """Show loading message to user."""
        print(f"\n🔄 Loading {model_type} model '{model_name}' for {purpose}")
        print("📦 This may take a few minutes on first run (downloading model)")
        print("💾 Models are cached locally for faster subsequent loads")
    
    def _show_success_message(self, model_type: str, model_name: str):
        """Show success message to user."""
        print(f"✅ {model_type} model '{model_name}' loaded successfully\n")
        logger.info(f"{model_type} model '{model_name}' loaded and cached")
    
    def clear_cache(self):
        """Clear the model cache to free memory."""
        logger.info("Clearing model cache")
        self._model_cache.clear()
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # Clear GPU cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_cache_info(self) -> Dict[str, Any]:
        """Get information about cached models."""
        return {
            "cached_models": list(self._model_cache.keys()),
            "cache_count": len(self._model_cache),
            "device": str(self.device),
            "cache_dir": str(self.cache_dir)
        }


# Global model loader instance
model_loader = ModelLoader()

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
LLAMA_8B = "meta-llama/Meta-Llama-3.1-8B"

def _load_model(model_name):
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Configure 8-bit quantization
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
    )

    # Load the model with quantization and efficient memory usage
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
        quantization_config=quantization_config,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )

    return model, tokenizer


def load_llama_8b():
    return _load_model(LLAMA_8B)
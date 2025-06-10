# Hugging Face authentication
import os
from huggingface_hub import login
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def hf_login():
    token = os.getenv("HUGGINGFACE_TOKEN")
    if not token:
        raise ValueError("HUGGINGFACE_TOKEN environment variable not set")
    login(token=token)

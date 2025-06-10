from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def get_openai_api_key():
    """Get OpenAI API key from environment variables."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable not set")
    return api_key

def load_gpt_35():
    api_key = get_openai_api_key()
    model = ChatOpenAI(api_key=api_key)
    return model

def load_gpt_4():
    api_key = get_openai_api_key()
    return ChatOpenAI(api_key=api_key, model="gpt-4")
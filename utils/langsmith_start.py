import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init():
    """Initialize LangSmith tracing with environment variables."""
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        raise ValueError("LANGCHAIN_API_KEY environment variable not set")
    
    os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
    os.environ["LANGCHAIN_API_KEY"] = api_key
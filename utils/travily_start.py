import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def init_tavily():
    """Initialize Tavily API with environment variables."""
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY environment variable not set")
    
    os.environ['TAVILY_API_KEY'] = api_key

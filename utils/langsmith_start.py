import os
API_KEY = "[REMOVED]"

def init():
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_API_KEY"] = API_KEY
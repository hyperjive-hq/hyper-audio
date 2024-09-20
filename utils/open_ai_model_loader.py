from langchain_openai import ChatOpenAI

API_KEY = "[REMOVED]"

def load_gpt_35():
    model = ChatOpenAI(api_key=API_KEY)
    return model
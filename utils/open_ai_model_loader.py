from langchain_openai import ChatOpenAI
import os

OPENAI_API_KEY = "[REMOVED]"
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

def load_gpt_35():
    model = ChatOpenAI(api_key=OPENAI_API_KEY)
    return model

def load_gpt_4():
    return ChatOpenAI(api_key=OPENAI_API_KEY, model="gpt-4")
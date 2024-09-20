# following https://python.langchain.com/docs/tutorials/chatbot/
from langchain_core.messages import HumanMessage

from utils.langsmith_start import init
from utils.open_ai_model_loader import load_gpt_35

init()
model = load_gpt_35()
response = model.invoke([HumanMessage(content="What's up you dirty dog")])
print(response)

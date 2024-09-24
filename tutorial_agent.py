# Import relevant functionality
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from utils.langsmith_start import init
from utils.travily_start import init_tavily
from utils.open_ai_model_loader import load_gpt_4


init()
init_tavily()

# Create the agent
memory = MemorySaver()
model = load_gpt_4()
search = TavilySearchResults(max_results=2)
tools = [search]



agent_executor = create_react_agent(model, tools)

for chunk in agent_executor.stream(
        {"messages": [HumanMessage(content="whats the weather in sf?")]}
):
    print(chunk)
    print("----")
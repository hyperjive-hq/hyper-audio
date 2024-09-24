# following https://python.langchain.com/docs/tutorials/chatbot/
from typing import List, Union

from langchain_core.messages import HumanMessage, AIMessage
from pydantic import Field, BaseModel

from utils.langsmith_start import init
from utils.open_ai_model_loader import load_gpt_35
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils.memory_session import get_session_history
from operator import itemgetter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.messages import SystemMessage, trim_messages
from langchain_core.output_parsers import StrOutputParser

from fastapi import FastAPI
from langserve import add_routes

init()
model = load_gpt_35()
session_config = {"configurable": {"session_id": "abc2"}}

prompt = ChatPromptTemplate([
        (
            "system",
            "You are a helpful hacker assistant. You are assuming you're a real person who is a computer hacker. You also love to use emoji!",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

parser = StrOutputParser()

chain = (
        RunnablePassthrough.assign(messages=itemgetter("messages") | trimmer)
        | prompt
        | model
        | parser
)

with_message_history = RunnableWithMessageHistory(chain, get_session_history, input_messages_key="messages")

class InputChat(BaseModel):
    """Input for the chat endpoint."""

    messages: List[Union[HumanMessage]] = Field(
        ...,
        description="The chat messages representing the current conversation.",
    )

app = FastAPI(
    title="Tutorial Memory Chatbot",
    version="1.0.0",
    description="Tutorial Memory Chatbot",
)
add_routes(
    app,
    chain.with_types(input_type=InputChat),
    path="/chain",
    enable_feedback_endpoint=True,
    enable_public_trace_link_endpoint=True)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)

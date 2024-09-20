#%%
# First, install the necessary libraries if you haven't already
# !pip install transformers torch accelerate
# [HF_TOKEN_REMOVED]
# meta-llama/Meta-Llama-3.1-8B

# Function to generate text with streaming
import torch
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from transformers import pipeline
from utils.local_model_loader import load_llama_8b

# Load the model and tokenizer
model, tokenizer = load_llama_8b()

# Create a pipeline with memory-efficient settings
pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=100,
    temperature=0.7,
    torch_dtype=torch.float16,
    device_map="auto",
)
# Create a HuggingFacePipeline LLM
llm = HuggingFacePipeline(pipeline=pipe)

# Create a prompt template
template = """Question: {question}"""

prompt = PromptTemplate.from_template(template)

# Create a chain
chain = (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
)

# Run the chain
question = "Can LLMs help execute tasks?"
response = chain.invoke(question)

print(response)
#%%

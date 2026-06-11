import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

load_dotenv()

# Initialize OpenAI model (recommended for beginners)
llm_openai = ChatOpenAI(
    model="gpt-4o",
    temperature=0, # More deterministic for agent behavior
    api_key=os.getenv("OPENAI_API_KEY")
)

llm = llm_openai
print("LLM initialized successfully")
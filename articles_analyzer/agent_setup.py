from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.tools import tool

# Load environment variables
load_dotenv()

# -----------------------------
# Define tools
# -----------------------------
@tool
def search(query: str) -> str:
    """Search for information."""
    return f"Search result for '{query}': LangChain is a framework for building LLM-powered applications."

@tool
def get_weather(location: str) -> str:
    """Get weather information for a location."""
    return f"The weather in {location} is sunny and 72°F."

tools = [search, get_weather]

# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)


# Bind tools to the model
tool_calling_agent = llm.bind_tools(tools)

# -----------------------------
# 1️⃣ First invoke (tool planning)
# -----------------------------
response = tool_calling_agent.invoke(
    "What is LangChain and what is the weather in New York?"
)

for call in response.tool_calls:
    print(call)
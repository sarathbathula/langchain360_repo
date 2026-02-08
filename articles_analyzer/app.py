# -----------------------------
# Imports
# -----------------------------
import sys
from typing import TypedDict, List

from dotenv import load_dotenv

from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

from langgraph.graph import StateGraph, END

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

# -----------------------------
# Define State (TypedDict = schema, NOT an object)
# -----------------------------
class State(TypedDict, total=False):
    text: str
    classification: str
    entities: List[str]
    summary: str


# -----------------------------
# Initialize LLM
# -----------------------------
llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
    #top_p=0.9
)

# -----------------------------
# Nodes
# -----------------------------
def classification_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Classify the following text into one of the categories: "
            "News, Blog, Research, or Other.\n\nText: {text}\n\nCategory:"
        )
    )

    message = HumanMessage(content=prompt.format(text=state["text"]))
    classification = llm.invoke([message]).content.strip()
    return {"classification": classification}


def entity_extraction_node(state: State):
    prompt = PromptTemplate(
        input_variables=["text"],
        template=(
            "Extract all entities (Person, Organization, Location) "
            "from the following text. Return a comma-separated list.\n\n"
            "Text: {text}\n\nEntities:"
        )
    )

    message = HumanMessage(content=prompt.format(text=state["text"]))
    entities_text = llm.invoke([message]).content.strip()

    entities = [e.strip() for e in entities_text.split(",") if e.strip()]

    return {"entities": entities}


def summarize_node(state: State):
    prompt = PromptTemplate.from_template(
        "Summarize the following text in one short sentence.\n\nText: {text}\n\nSummary:"
    )

    chain = prompt | llm
    response = chain.invoke({"text": state["text"]})

    return {"summary": response.content.strip()}


# -----------------------------
# Build LangGraph Workflow
# -----------------------------
workflow = StateGraph(State)

workflow.add_node("classification", classification_node)
workflow.add_node("entity_extraction", entity_extraction_node)
workflow.add_node("summarization", summarize_node)

workflow.set_entry_point("classification")
workflow.add_edge("classification", "entity_extraction")
workflow.add_edge("entity_extraction", "summarization")
workflow.add_edge("summarization", END)

app = workflow.compile()

# -----------------------------
# Test
# -----------------------------
# sample_text = """
# Anthropic's MCP (Model Context Protocol) is an open-source framework
# that allows applications to interact with multiple external systems.
# """

sample_text = """
    Apple announced on Tuesday that it will launch its next-generation iPhone in September 2026, featuring a new AI-powered camera system and improved battery life.
"""

# ✅ State is just a dict matching TypedDict schema
state_input: State = {
    "text": sample_text
}

result = app.invoke(state_input)

print("Input Text:", sample_text )
print("Classification:", result["classification"])
print("Entities:", result["entities"])
print("Summary:", result["summary"])

sys.exit(0)

# Build Chatbot using OpenAI and Langchain framework 
from dotenv import load_dotenv
import os 
import openai 


from datetime import datetime
from pydantic import BaseModel, field_validator

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_openai import ChatOpenAI

# load env 
load_dotenv()

# openai key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# -----------------------------
# 1. Output Schema (Deterministic)
# -----------------------------
class DateResult(BaseModel):
    event_time: datetime

    @field_validator("event_time")
    @classmethod
    def must_be_iso8601(cls, v: datetime):
        # enforce timezone-aware datetime if required
        if v.tzinfo is None:
            raise ValueError("Datetime must include timezone information")
        return v


# -----------------------------
# 2. Output Parser
# -----------------------------
parser = PydanticOutputParser(pydantic_object=DateResult)


# -----------------------------
# 3. Prompt Template
# -----------------------------
prompt = PromptTemplate(
    template="""
        Extract the event datetime from the text below.

        Text:
        {text}

        Rules:
        - Convert to ISO-8601 format
        - Include timezone (assume UTC if not specified)

        {format_instructions}
        """,
            input_variables=["text"],
            partial_variables={
                "format_instructions": parser.get_format_instructions()
            },
    )


# -----------------------------
# 4. LLM (SLM / LLM)
# -----------------------------
# chat object 
llm = ChatOpenAI(model_name = 'gpt-4o-mini', 
                seed = 365,
                temperature = 0, 
                max_tokens = 100
                )

# -----------------------------
# 5. Chain
# -----------------------------
chain = prompt | llm | parser


# -----------------------------
# 6. Invocation
# -----------------------------
result = chain.invoke(
    {
        "text": "The patient follow-up is scheduled on March 12, 2025 at 10:30 AM"
    }
)

print("Parsed datetime:", result.event_time)
print("Type:", type(result.event_time))

# build chat bot application using langchain framework and openai api platform 
import os
from dotenv import load_dotenv

from langchain_openai import ChatOpenAI

from langchain_core.prompts import ChatPromptTemplate 
from langchain_core.output_parsers import CommaSeparatedListOutputParser

# load env
load_dotenv()

# openai_api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#initiate chat client
chat = ChatOpenAI(model_name = 'gpt-4o-mini',
                  seed = 365,
                  temperature = 0,
                  max_tokens = 100 )


# list instructions 
list_instructions = CommaSeparatedListOutputParser().get_format_instructions()

#chat template
chat_template = ChatPromptTemplate.from_messages(('human', "I've recently adopted a {pet}. Can you suggest three {pet} names ? \n" + list_instructions))

#print(chat_template.messages[1].prompt.template)


# output parser
list_output_parser = CommaSeparatedListOutputParser()

# passing actual values to chat template
chat_template_result = chat_template.invoke({'pet' : 'dog'})

# respnse = chat.invoke(chat_template_result)
# print(respnse)

# # Expected result
# expected_result = list_output_parser.invoke(respnse)
# print(expected_result)

# one liner using LCEL 
chain = chat_template | chat | list_output_parser

# pass pet name 
result = chain.invoke({'pet' : 'dog'})

print(result)

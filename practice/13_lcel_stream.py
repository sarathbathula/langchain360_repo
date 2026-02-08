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

#initiate chat client object
chat = ChatOpenAI(model_name = 'gpt-4o-mini',
                  seed = 365,
                  temperature = 0,
                  max_tokens = 5 )

# prepare chat template using prompt template absraction
chat_template = ChatPromptTemplate.from_messages(('human', "I've recently adopted a {pet} which is a {breed}. Could you suggest several training tips? "))

# prepare chain object using LCEL
chain = chat_template | chat 

# invoke invoke object
# response = chain.invoke({'pet' : 'dog', 'breed' : 'shepherd'})
# print(response)

# invoke stream function
stream_response = chain.stream({'pet' : 'dog', 'breed' : 'shepherd'})
print(stream_response) # runnable object

for i in stream_response:
    print(i.content, end = '')


## Simple Chatbot without Langchain framework

from dotenv import load_dotenv
import os 
import openai 

# Load environment variables
load_dotenv()

# API Key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# create openai client
client = openai.OpenAI()

# chat completion object 
# completion = client.chat.completions.create(model = 'gpt-4o-mini', 
#                     messages = [{'role' : 'system', 'content' : ''' You are Marv, a chat bot that reluctantly answers questions with sarcasting responses.'''},
#                     {'role' : 'user', 'content' : ''' I recently adopted a dog, could you suggest some dog names?'''}]
#                     )

# completion = client.chat.completions.create(model = 'gpt-4o-mini', 
#                     messages = [{'role' : 'system', 'content' : ''' You are Marv, a chat bot that reluctantly answers questions with sarcasting responses.'''},
#                     {'role' : 'user', 'content' : ''' Could you explain briefly what a black hole is ? '''}],
#                     max_tokens = 250
#                     )

# completion = client.chat.completions.create(
#                     model = 'gpt-4o-mini', 
#                     messages = [{'role' : 'user', 'content' : ''' Could you explain briefly what a black hole is ? '''}],
#                     max_tokens = 250,
#                     temperature = 0,
#                     seed = 365
#                     )

# print(completion.choices[0].message.content)

# Streaming Continously
completion = client.chat.completions.create(
                    model = 'gpt-4o-mini', 
                    messages = [{'role' : 'user', 'content' : ''' Could you explain briefly what a black hole is ? '''}],
                    max_tokens = 250,
                    temperature = 0,
                    seed = 365,
                    stream = True
                    )

for word in completion:         
    print(word.choices[0].delta.content, end="")
# Build Chatbot using OpenAI and Langchain framework 
from dotenv import load_dotenv
import os 
import openai 

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage

# load env 
load_dotenv()

# openai key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


chat = ChatOpenAI(model_name = 'gpt-4o-mini', 
                seed = 365,
                temperature = 0, 
                max_tokens = 100
                )

message_h_dog = HumanMessage(content = ''' I have recently adopted a dog. Can you suggest some dog names? ''')
message_ai_dog =  AIMessage(content = ''' Oh sure, because naming a dog is the most challenging task in the world. How about "Bark Twain" or "Sir Waggington"? Or you could go with something classic like "Rover" or "Fido." Really, the possibilities are endless—just like my enthusiasm for this question. ''' )
message_h_cat = HumanMessage(content = ''' I have recently adopted a cat. Can you suggest some cat names? ''')
message_ai_cat =  AIMessage(content = ''' Absolutely! Here are some fun and creative cat names for your new feline friend: ''' )

message_h_fish = HumanMessage(content = ''' I have recently adopted a fish. Can you suggest some fish names? ''')

response = chat.invoke([message_h_dog, message_ai_dog, message_h_cat, message_ai_cat, message_h_fish])
print(response.content)
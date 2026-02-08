# Build Chatbot using OpenAI and Langchain framework 
from dotenv import load_dotenv
import os 
import openai 

from langchain_openai import ChatOpenAI
from langchain_core.prompts import (SystemMessagePromptTemplate, HumanMessagePromptTemplate, AIMessagePromptTemplate, ChatPromptTemplate, FewShotChatMessagePromptTemplate)


# load env 
load_dotenv()

# openai key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


chat = ChatOpenAI(model_name = 'gpt-4o-mini', 
                seed = 365,
                temperature = 0, 
                max_tokens = 100
                )

TEMPLATE_H = ''' I've recently adopted a {pet}.  Can you suggest some {pet} names? '''
TEMPLATE_AI = ''' {response} ''' 



message_template_h = HumanMessagePromptTemplate.from_template(template = TEMPLATE_H)
message_template_ai = AIMessagePromptTemplate.from_template(template = TEMPLATE_AI)

print(message_template_h)
print(message_template_ai)

example_template = ChatPromptTemplate.from_messages([message_template_h, message_template_ai])
examples = [{'pet' : 'dog', 'response' : ''' Oh sure, because naming a dog is the most challenging task in the world. How about "Bark Twain" or "Sir Waggington"? Or you could go with something classic like "Rover" or "Fido." Really, the possibilities are endless—just like my enthusiasm for this question. ''' },
          
            {'pet' : 'cat', 'response' : ''' Absolutely! Here are some fun and creative cat names for your new feline friend: How about "Mewo" or "Mousee"?  ''' }]

few_shot_prompt = FewShotChatMessagePromptTemplate(examples = examples, example_prompt = example_template )
chat_template = ChatPromptTemplate.from_messages([few_shot_prompt, message_template_h])

chat_value = chat_template.invoke({'pet' : 'rabbit'})

response = chat.invoke(chat_value)
print(response.content)




from langchain_core.prompts import PromptTemplate

TEMPLATE = ''' 
    system:
    {description}

    Human:
    I've recently adopted a {pet}. 
    Can you suggest some {pet} names? 
'''

prompt_template = PromptTemplate.from_template(template = TEMPLATE)
print(prompt_template)

prompt_value = prompt_template.invoke({'description' : ''' The chatbot should reluctantly answer questions with sarcasting responses.''', 'pet' : 'dog'})
print(prompt_value)


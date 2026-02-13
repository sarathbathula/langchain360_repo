# Maximum Marigin Relevance Search

import os
from dotenv import load_dotenv

from langchain_openai.embeddings import OpenAIEmbeddings 
from langchain_chroma import Chroma
from langchain_core.documents import Document

# load env
load_dotenv()

# openai_api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#define embedding
embedding = OpenAIEmbeddings(model = "text-embedding-ada-002")

# read from direcotory
vectorstore = Chroma(persist_directory =  "../embedding_directory", embedding_function = embedding)

added_document = Document(page_content='Alright! So… How are the techniques used in data, business intelligence, or predictive analytics applied in real life? Certainly, with the help of computers. You can basically split the relevant tools into two categories—programming languages and software. Knowing a programming language enables you to devise programs that can execute specific operations. Moreover, you can reuse these programs whenever you need to execute the same action', 
                          metadata={'Course Title': 'Introduction to Data and Data Science', 
                                    'Lecture Title': 'Programming Languages & Software Employed in Data Science - All the Tools You Need'})

id = vectorstore.add_documents([added_document])
print(id)

question = "What software do data scientists use?"

retreived_documents = vectorstore.max_mariginal_relavance_search(query = question, k = 3, lambad_mult = 1, filter = {"Lecture Title": "Programming Languages & Software Employed in Data Science - All the Tools You Need"})
print(retreived_documents)

for i in retreived_documents:
    print(f"Page Content: {i.page_content}\n----------\nLecture Title:{i.metadata['Lecture Title']}\n")


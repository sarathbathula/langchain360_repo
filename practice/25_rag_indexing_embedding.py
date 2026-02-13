import os
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader 
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter 
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
import numpy as np

loader_docx = Docx2txtLoader("../data/Introduction_to_Data_and_Data_Science2.docx")
pages = loader_docx.load()

md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Course Title") , ("##", "Lecture Title")])
pages_md_split = md_splitter.split_text(pages[0].page_content)

for i in range(len(pages_md_split)):
    pages_md_split[i].page_content = ' '.join(pages_md_split[i].page_content.split())

char_splitter = CharacterTextSplitter( 
        separator = ".",
        chunk_size = 500,
        chunk_overlap = 50
)

pages_char_split = char_splitter.split_documents(pages_md_split)


# load env
load_dotenv()

# openai_api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#initiate chat client object
# chat = ChatOpenAI(model_name = 'gpt-4o-mini',
#                   seed = 365,
#                   temperature = 0,
#                   max_tokens = 100 )

embedding = OpenAIEmbeddings(model = "text-embedding-ada-002")

print(pages_char_split[18].page_content)

vector1 = embedding.embed_query(pages_char_split[3].page_content)
vector2 = embedding.embed_query(pages_char_split[5].page_content)
vector3 = embedding.embed_query(pages_char_split[18].page_content)

print(len(vector1), len(vector2), len(vector3))

print(np.dot(vector1, vector2))
print(np.dot(vector1, vector3))
print(np.dot(vector2, vector3))

print("....")

print(np.linalg.norm(vector1))
print(np.linalg.norm(vector2))
print(np.linalg.norm(vector3))





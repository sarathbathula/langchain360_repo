import os
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader 
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter 
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma

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
print(len(pages_char_split))

# load env
load_dotenv()

# openai_api_key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

#define embedding
embedding = OpenAIEmbeddings(model = "text-embedding-ada-002")

# vectorstore
vectorstore = Chroma.from_documents(documents = pages_char_split, embedding = embedding, persist_directory = "../embedding_directory")

vectorstore_from_directory = Chroma(persist_directory =  "../embedding_directory", embedding_function = embedding)






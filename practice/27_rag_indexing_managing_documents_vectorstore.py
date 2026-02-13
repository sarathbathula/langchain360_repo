import os
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader 
from langchain_text_splitters.markdown import MarkdownHeaderTextSplitter 
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_chroma import Chroma
from langchain_core.documents import Document

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

# vectorstore from db
vectorstore_from_directory = Chroma(persist_directory =  "../embedding_directory", embedding_function = embedding)

print(vectorstore_from_directory.get(ids='22c091cf-ac0d-45f1-a865-893b453a0f8e', include = ['embeddings']))

added_document = Document(page_content='Alright! So… Let’s discuss the not-so-obvious differences between the terms analysis and analytics. Due to the similarity of the words, some people believe they share the same meaning, and thus use them interchangeably. Technically, this isn’t correct. There is, in fact, a distinct difference between the two. And the reason for one often being used instead of the other is the lack of a transparent understanding of both. So, let’s clear this up, shall we? First, we will start with analysis', 
                          metadata={'Course Title': 'Introduction to Data and Data Science', 
                                    'Lecture Title': 'Analysis vs Analytics'})

id = vectorstore_from_directory.add_documents([added_document])
print(id)

print(vectorstore_from_directory.get("2742d750-7724-4ba0-a210-006ec1eb20f0"))

updated_document = Document(page_content='Great! We hope we gave you a good idea about the level of applicability of the most frequently used programming and software tools in the field of data science. Thank you for watching!', 
                            metadata={'Course Title': 'Introduction to Data and Data Science', 
                                     'Lecture Title': 'Programming Languages & Software Employed in Data Science - All the Tools You Need'})

vectorstore_from_directory.update_document(document_id = "2742d750-7724-4ba0-a210-006ec1eb20f0", 
                                           document = updated_document)

print(vectorstore_from_directory.get("2742d750-7724-4ba0-a210-006ec1eb20f0"))

print("Deleting a document")
vectorstore_from_directory.delete("2742d750-7724-4ba0-a210-006ec1eb20f0")
print(vectorstore_from_directory.get("2742d750-7724-4ba0-a210-006ec1eb20f0"))
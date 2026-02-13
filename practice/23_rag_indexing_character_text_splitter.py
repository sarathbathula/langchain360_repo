from langchain_community.document_loaders import Docx2txtLoader 
from langchain_text_splitters.character import CharacterTextSplitter

pages_docx = Docx2txtLoader("../data/Introduction_to_Data_and_Data_Science.docx")
pages = pages_docx.load()

for i in range(len(pages)):
    pages[i].page_content = ' '.join(pages[i].page_content.split())

print(len(pages[0].page_content))

char_spiltter = CharacterTextSplitter(separator = ".", chunk_size = 500, chunk_overlap = 50)
pages_char_split = char_spiltter.split_documents(pages)

print(pages_char_split)
print(len(pages_char_split))



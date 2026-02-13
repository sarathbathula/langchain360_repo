from langchain_community.document_loaders import Docx2txtLoader 



loader_docx = Docx2txtLoader("../data/Introduction_to_Data_and_Data_Science.docx")

pages_docx = loader_docx.load()

print(pages_docx)

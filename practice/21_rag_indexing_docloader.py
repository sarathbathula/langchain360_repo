from langchain_community.document_loaders import PyPDFLoader
import copy 

loader_pdf = PyPDFLoader("../data/Introduction_to_Data_and_Data_Science.pdf")

pages_pdf = loader_pdf.load()

pages_pdf_cut = copy.deepcopy(pages_pdf)


for page in pages_pdf_cut:
    page.page_content = ' '.join(page.page_content.split())

print(pages_pdf_cut[0].page_content)
print(pages_pdf[0].page_content)




import os 
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document
import fitz
from PIL import Image
st.set_page_config(page_title="PDF Q qnd A Example",layout="wide")
st.title("UPLOAD an PDF and create a chatbot ")
uploaded_file = st.file_uploader("upload a PDF file",type=["pdf"])
if uploaded_file:
    pdf_bytes = uploaded_file.read()
    with open("temp.pdf","wb") as f:
        f.write(pdf_bytes)
    
    loader = PyPDFLoader("temp.pdf")
    pages = loader.load()
    st.success(f"Loaded{len(pages)} pages")
    splitter = RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=100)
    docs = []
    for page in pages:
        splits = splitter.split_text(page.page_content)
        for chunk in splits:
            docs.append(Document(page_content=chunk,metadata={"page":page.metadata["page"]}))
    st.info(f"split into {len(docs)} chunks")
    #create Embeddings
    embeddings = OpenAIEmbeddings(openai_api_key = OPENAI_API_KEY)
    with st.spinner("Creating vector Embeddings ..."):
        vectorstore = FAISS.from_documents(docs, embeddings)
    st.success("Vector store created using openai embdeddings")
    query = st.text_input("Ask something about the given pdf")
    if query:
        results = vectorstore.similarity_search(query,k=3)
        st.markdown("###top matches")
        doc = fitz.open("temp.pdf")
        for i ,res in enumerate(results,1):
            page_num = res.metadata.get("page",0)
            text_to_highlight = res.page_content.strip()
            st.markdown(f"**Result {i} (page {page_num+1} ):**")
            st.write(text_to_highlight)
            #hilight pdf 
            page = doc.load_page(page_num)
            highlight_instances = page.search_for(text_to_highlight)
            for inst in highlight_instances:
                page.add_highlight_annot(inst)
            #render the page
            pix = page.get_pixmap(dpi=150)
            img_path = f"highlighted page_{page_num +1}.png"
            pix.save(img_path)
            st.image(Image.open(img_path),caption=f"highloghted page{page_num+1}",use_column_width=True)
    
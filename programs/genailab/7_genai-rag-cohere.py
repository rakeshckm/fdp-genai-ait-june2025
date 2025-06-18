# pip install streamlit PyPDF2 requests langchain langchain-community langchain-cohere cohere python-dotenv faiss-cpu

import os
import requests
import PyPDF2
from dotenv import load_dotenv
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_cohere import CohereEmbeddings, ChatCohere

# Load environment variables from .env
load_dotenv()

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PyPDF2.PdfReader(file)
        text = "\n".join([page.extract_text() for page in reader.pages if page.extract_text()])
    return text

def download_and_extract_ipc():
    pdf_path = "IPC_186045.pdf"

    if not os.path.exists(pdf_path):
        url = "https://www.indiacode.nic.in/repealedfileopen?rfilename=A1860-45.pdf"
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(pdf_path, "wb") as file:
                for chunk in response.iter_content(chunk_size=1024):
                    file.write(chunk)
            st.success("IPC PDF downloaded successfully.")
        else:
            st.error("Failed to download the IPC PDF.")
            return None

    text = extract_text_from_pdf(pdf_path)

    if text.strip():
        with open("Indian_Penal_Code.txt", "w", encoding="utf-8") as f:
            f.write(text)
        return text
    else:
        st.error("No extractable text found in the PDF.")
        return None

@st.cache_resource(show_spinner="Embedding IPC text...")
def create_retriever(text):
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(text)
    embeddings = CohereEmbeddings(model="embed-english-v3.0")  # You can customize the model
    vector_store = FAISS.from_texts(texts, embeddings, normalize_L2=True)
    return vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

def main():
    st.title("📘 IPC Chatbot (Cohere Powered)")
    st.write("Ask anything about the Indian Penal Code (1860).")

    cohere_key = os.getenv("COHERE_API_KEY")
    if not cohere_key:
        st.error("Cohere API key not found. Please set it in your .env file.")
        return

    if "retriever" not in st.session_state:
        text = download_and_extract_ipc()
        if not text:
            return
        retriever = create_retriever(text)
        st.session_state.retriever = retriever

    llm = ChatCohere(model="command-r-plus", temperature=0.3)

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=st.session_state.retriever)

    user_input = st.text_input("Enter your query:")
    if user_input:
        with st.spinner("Searching..."):
            response = qa_chain.invoke(user_input)
            st.markdown("**Bot:** " + response["result"])

if __name__ == "__main__":
    main()

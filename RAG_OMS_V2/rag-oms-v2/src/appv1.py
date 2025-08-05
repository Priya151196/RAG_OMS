from pypdf import PdfReader
from langchain.docstore.document import Document
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI
from time import sleep
import os
from config import *
import asyncio
import streamlit as st
import tempfile
from config import *
from langchain_community.vectorstores import FAISS

st.set_page_config(page_title="RAG based OMS Assistant powered by LangChain and Google Gemini", layout="wide")


def get_document_text(uploaded_file=None, dataFolderPath = DATA_DIR, title=None):
    if uploaded_file:
        docs = []
        fname = uploaded_file.name
        if not title:
            title = os.path.basename(fname)
        if fname.lower().endswith('pdf'):
            pdf_reader = PdfReader(uploaded_file)
            for num, page in enumerate(pdf_reader.pages):
                page = page.extract_text()
                doc = Document(page_content=page, metadata={'title': title, 'page': (num + 1)})
                docs.append(doc)
    else:
        documentLoader = PyPDFDirectoryLoader(dataFolderPath)
        docs = documentLoader.load()
    return docs


def textSplitter(docs):
    rawData = "\n".join([d.page_content for d in docs])
    textSplitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    splitData = textSplitter.create_documents([rawData])
    return splitData


# This is to get the Streamlit app to use less CPU while embedding documents into Chromadb.
class EmbeddingProxy:
    def __init__(self, embedding, use_delay=True):
        self.embedding = embedding
        self.use_delay = use_delay

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.use_delay:
            sleep(EMBED_DELAY)
        return self.embedding.embed_documents(texts)

    def embed_query(self, text: str) -> list[float]:
        if self.use_delay:
            sleep(EMBED_DELAY)
        return self.embedding.embed_query(text)


@st.cache_resource(show_spinner="Loading vector database...")
def create_vector_db(_texts, embeddings=None):

    # Initialize embeddings if not provided
    if not embeddings:
        embeddings = GoogleGenerativeAIEmbeddings(
            model=EMBEDDING_MODEL_NAME,
            task_type="retrieval_document"
        )
    proxy_embeddings = EmbeddingProxy(embeddings, use_delay=False)

    # FAISS is always built in memory due to ephemeral storage on Streamlit Cloud
    print("Creating FAISS vector DB in memory (no persistent storage).")
    db = FAISS.from_texts(_texts, proxy_embeddings)

    return db


def ragChain(question):
    docs = get_document_text(dataFolderPath = DATA_DIR)
    splitData = textSplitter(docs)
    vs = create_vector_db(splitData)
    retriever = vs.as_retriever()
    context = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in context)
    output_parser = StrOutputParser()
    llm = ChatGoogleGenerativeAI(model=LLM_NAME, convert_system_message_to_human=True)
    chain = prompt | llm | output_parser
    response = chain.invoke({'question' :  question, 'context' : context})
    return response, vs

import re

def clean_collection_name(name: str) -> str:
    # Remove invalid characters
    name = re.sub(r'[^a-zA-Z0-9._-]', '', name)
    # Ensure valid start and end characters
    name = re.sub(r'^[^a-zA-Z0-9]+', '', name)
    name = re.sub(r'[^a-zA-Z0-9]+$', '', name)
    # Enforce length constraints
    return name[:512] if len(name) >= 3 else name + "_01"

# ----------------- Custom CSS Styling ----------------- #
def apply_custom_styles():
    st.markdown("""
        <style>
            .main {
                background-color: #f9f9fc;
                font-family: 'Segoe UI', sans-serif;
            }
            h1 {
                color: #2a2e82;
                font-weight: 700;
            }
            .sidebar .sidebar-content {
                background-color: #e6e9ff;
                padding: 1rem;
            }
            .uploadedFile {
                color: #2a2e82;
                font-weight: 600;
            }
            .stTextInput > label {
                font-weight: 700;
                font-size: 18px !important;
            }
            .message {
                padding: 0.5rem;
                border-radius: 8px;
                margin-bottom: 10px;
            }
            .user {
                background-color: #d9e8ff;
                text-align: right;
            }
            .bot {
                background-color: #fff5d7;
            }
        </style>
    """, unsafe_allow_html=True)


# ----------------- App UI ----------------- #
def main():
    apply_custom_styles()

    
    st.title("📚 RAG based OMS Assistant Powered by Gemini")

    with st.sidebar:
        st.header("📤 Please upload PDF document")
        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")
    with st.sidebar:
        st.header("🔑 Gemini API Key")
        geminiKey = st.text_input(
        label="Please provide Gemini API key👇",
        )   

    os.environ["GOOGLE_API_KEY"] = geminiKey

        
    if uploaded_file:
        current_file_name = uploaded_file.name
        if st.session_state.get("uploaded_file_name") != current_file_name:
            # New file uploaded — reset vectorstore and chat history
            st.session_state["uploaded_file_name"] = current_file_name
            # st.session_state["vs"] = None
            st.session_state["chat_history"] = []

    

    if uploaded_file:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        st.success("✅ PDF Uploaded Successfully!")


        st.markdown("#### Ask a question based on your document 👇")
        query = st.text_input("🧑‍💻 Type your query")

    

        if "vector_db" not in st.session_state:
            if uploaded_file:
                if query :
                    with st.spinner("Retrieving answer..."):
                        if "vs" not in st.session_state:
                            if uploaded_file:
                                docs = get_document_text(uploaded_file)
                                split_docs = textSplitter(docs)
                                collectionName = clean_collection_name(os.path.splitext(uploaded_file.name)[0].replace(" ", "_"))
                                st.session_state.vs = create_vector_db(split_docs, collection_name=collectionName)
                        vs = st.session_state.vs
                        retriever = vs.as_retriever()
                        context = retriever.invoke(query)
                        context = "\n\n".join(doc.page_content for doc in context)
                        output_parser = StrOutputParser()
                        llm = ChatGoogleGenerativeAI(model=LLM_NAME, convert_system_message_to_human=True)
                        chain = prompt | llm | output_parser
                        answer = chain.invoke({'question' :  query, 'context' : context})


                    # st.markdown(f'<div class="message user">{query}</div>', unsafe_allow_html=True)
                    # st.markdown(f'<div class="message bot">{answer}</div>', unsafe_allow_html=True)
                    # Display user query with icon
                    # st.markdown(f'''
                    #     <div class="message user">
                    #         <span style="margin-right: 8px;">🧑‍💻</span>{query}
                    #     </div>
                    # ''', unsafe_allow_html=True)

                    # Display bot response with icon
                    st.markdown(f'''
                    <div class="message bot">
                        <span style="font-size: 28px; margin-right: 10px; color: #f59e0b;">🤖</span>
                        <span style="font-size: 16px;">{answer}</span>
                    </div>
                ''', unsafe_allow_html=True)




if __name__ == "__main__":
    main()

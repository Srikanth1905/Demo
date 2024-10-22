import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import (
    DirectoryLoader,
    TextLoader,
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredMarkdownLoader
)
import warnings

# Configuration and Setup
load_dotenv()
st.set_page_config(page_title="DocuMate: LLM-RAG Assistant", page_icon="🤖", layout="wide")
st.title("DocuMate: LLM-RAG Assistant 🤖")

# Suppress warnings
warnings.filterwarnings("ignore")
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize paths and session state
data = os.getenv('DATA_PATH')

if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "embeddings" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

@st.cache_resource
def initialize_llm():
    return OllamaLLM(
        model=os.getenv("OLLAMA_MODEL"),
        temperature=0.1
    )

def create_enhanced_rag_prompt():
    return ChatPromptTemplate.from_template("""
        You are an expert AI assistant tasked with providing comprehensive answers using **only** the context provided.
        
        **Instructions:**
        1. Carefully read the question and the provided context.
        2. If the context is sufficient, generate an accurate, relevant, and structured answer.
        3. If the context lacks necessary information, clearly state that additional details are required.

        **Context**: {context}
        **Question**: {input}
        
        **Your Response**:
        1. **Direct Answer**: Provide a precise and concise answer.
        2. **Supporting Details**: Expand with relevant details from the context.
        3. **Limitations**: Note any uncertainties if the context is incomplete.
    """)

def setup_retrieval_chain():
    llm = initialize_llm()
    prompt = create_enhanced_rag_prompt()
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    retriever = st.session_state.vectors.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={
            "top_k": 7,
            "score_threshold": 0.2
        }
    )
    
    return create_retrieval_chain(retriever, document_chain)

# File uploader in sidebar
with st.sidebar:
    st.header("Document Processing")
    
    uploaded_files = st.file_uploader(
        "Upload your documents",
        accept_multiple_files=True,
        type=['txt', 'pdf', 'docx', 'md']
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            file_extension = os.path.splitext(uploaded_file.name)[1].lower()
            if file_extension in ['.txt', '.pdf', '.docx', '.md']:
                with open(os.path.join(data, uploaded_file.name), "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.success(f"File {uploaded_file.name} has been uploaded successfully!")
            else:
                st.warning(f"Unsupported file format: {file_extension}")

    if st.button("Process Documents"):
        start_time = time.time()
        all_docs = []

        loaders = [
            DirectoryLoader(data, glob="**/*.txt", loader_cls=TextLoader, use_multithreading=True, loader_kwargs={'encoding': 'utf-8'}),
            DirectoryLoader(data, glob="**/*.docx", loader_cls=UnstructuredWordDocumentLoader, use_multithreading=True),
            DirectoryLoader(data, glob="**/*.md", loader_cls=UnstructuredMarkdownLoader, use_multithreading=True),
            DirectoryLoader(data, glob="**/*.pdf", loader_cls=PyMuPDFLoader, use_multithreading=True),
        ]

        with st.spinner("Loading documents..."):
            for loader in loaders:
                try:
                    all_docs.extend(loader.load())
                except Exception as e:
                    st.warning(f"Failed to load some documents: {e}")

            if all_docs:
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=250)
                final_documents = text_splitter.split_documents(all_docs)
                
                st.session_state.vectors = FAISS.from_documents(final_documents, st.session_state.embeddings)
                total_time = time.time() - start_time
                st.success(f"Documents processed successfully in {total_time:.2f} seconds!")
            else:
                st.warning("No documents found to process.")

    if st.button("Reset All"):
        st.session_state.vectors = None
        st.session_state.messages = []
        st.success("All data has been reset!")

# Main chat interface
chat_container = st.container()
with chat_container:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

# Chat input and response generation
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    if not st.session_state.vectors:
        st.warning("Please process some documents first!")
    else:
        with st.chat_message("user"):
            st.markdown(user_question)
        st.session_state.messages.append({"role": "user", "content": user_question})
        
        try:
            with st.spinner("Generating response..."):
                start_time = time.time()
                retrieval_chain = setup_retrieval_chain()
                response = retrieval_chain.invoke({'input': user_question})
                response_time = time.time() - start_time
                
                answer = response['answer']
                with st.chat_message("assistant"):
                    st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
                st.info(f"Response generated in {response_time:.2f} seconds")
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try rephrasing your question or checking if the documents are properly processed.")

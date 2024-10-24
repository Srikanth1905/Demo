import os
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
    PyMuPDFLoader,
    TextLoader,
    Docx2txtLoader
)
import glob
import warnings
import tempfile
import time

# Configuration and Setup
load_dotenv()
st.set_page_config(page_title="LLM RAG Assistant", page_icon="🤖", layout="wide")
st.title("LLM RAG Assistant 🤖")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="langchain_community")
warnings.filterwarnings("ignore", module="langchain")
warnings.filterwarnings("ignore", module="langchain_ollama")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
LOCAL_DOCS_DIR = "./data"  # Change this to your desired local directory

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []
ollama_model = os.getenv('OLLAMA_MODEL')
@st.cache_resource
def initialize_llm():
    return OllamaLLM(model='llama3.2')

@st.cache_resource
def initialize_embeddings():
    start = time.time()
    return HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2',model_kwargs={'device': 'cpu'})

def get_file_loader(file_path):
    """Get appropriate loader based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {".pdf": PyMuPDFLoader,".docx": Docx2txtLoader,".txt": TextLoader}
    return loaders.get(ext)

def load_single_file(file_path):
    """Load a single file using appropriate loader"""
    loader_class = get_file_loader(file_path)
    if loader_class:
        try:
            if loader_class == TextLoader:
                loader = loader_class(file_path, encoding="utf8")
            else:
                loader = loader_class(file_path)
            return loader.load()
        except Exception as e:
            st.sidebar.error(f"Error loading {file_path}: {str(e)}")
            return []
    return []

def load_local_directory(directory_path):
    """Load documents from local directory"""
    documents = []
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        st.sidebar.info(f"Created directory: {directory_path}")
        return documents

    with st.spinner("Loading local documents..."):
        for ext in [".pdf", ".docx", ".txt"]:
            pattern = os.path.join(directory_path, f"*{ext}")
            files = glob.glob(pattern)
            for file_path in files:
                docs = load_single_file(file_path)
                if docs:
                    documents.extend(docs)
                    st.sidebar.write(f"Loaded: {os.path.basename(file_path)}")
    
    return documents

@st.cache_data
def process_documents(documents):
    """Process and split documents"""
    if not documents:
        return []
        
    with st.spinner("Processing documents..."):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=3000,chunk_overlap=750,length_function=len,add_start_index=True)
        splits = text_splitter.split_documents(documents)
        st.sidebar.write(f"Created {len(splits)} chunks from {len(documents)} documents")
        
        return splits

def create_rag_prompt():
    return ChatPromptTemplate.from_template("""
You are an AI assistant focused on providing information in the given context. You should only give factual information from the provided context.

**Instructions**:
1. Don't Use your pre trained knowledge.
2. Repsond with factual and accurate information.
3. Focus on summarizing the context, providing relevant insights, and highlighting the exact details present in the context.
                                            
**Context**: {context}
**Question**: {input}

**Your Response**:
1. **Direct Answer**: Provide a concise, accurate answer.
2. **Relevant Analysis**: Analyze the context and give summary. 
""")


def setup_retrieval_chain():
    start = time.time()
    llm = initialize_llm()
    prompt = create_rag_prompt()
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever(search_type = "similarity", search_kwargs = {"k":2,"score_threshold":0.7})
    return create_retrieval_chain(retriever, document_chain)

def process_uploaded_files(uploaded_files):
    """Process uploaded files and return documents"""
    documents = []
    with tempfile.TemporaryDirectory() as temp_dir:
        for uploaded_file in uploaded_files:
            # Save uploaded file to temporary directory
            temp_path = os.path.join(temp_dir, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            # Load and process the file
            docs = load_single_file(temp_path)
            if docs:
                documents.extend(docs)
                st.session_state.processed_files.append(uploaded_file.name)
    return documents

# Sidebar for document processing
with st.sidebar:
    st.header("Document Processing")
    
    # File upload option
    st.subheader("Upload Files")
    uploaded_files = st.file_uploader("Upload your documents",type=["pdf", "docx", "txt"],accept_multiple_files=True)
    
    # Local directory option
    st.subheader("Local Documents")
    st.write(f"Reading from: {LOCAL_DOCS_DIR}")
    use_local = st.checkbox("Include local documents")
    
    process_button = st.button("Process Documents")
    
    if process_button:
        documents = []
        
        # Process uploaded files
        if uploaded_files:
            upload_docs = process_uploaded_files(uploaded_files)
            documents.extend(upload_docs)
        
        # Process local directory if checked
        if use_local:
            local_docs = load_local_directory(LOCAL_DOCS_DIR)
            documents.extend(local_docs)
        
        if documents:
            splits = process_documents(documents)
            if splits:
                with st.spinner("Creating vector store..."):
                    start = time.time()
                    embeddings = initialize_embeddings()
                    st.session_state.vectors = FAISS.from_documents(splits, embeddings)
                    vector_creation_time = time.time() - start
                    st.info(f"Vector Store created in {vector_creation_time:.2f} seconds")
                    print(f"Vector Store created in {vector_creation_time:.2f} seconds")
                    st.success("Documents processed successfully! Ready for questions.")
        else:
            st.warning("No documents to process. Please upload files or select local documents.")
    
    # Display processed files
    if st.session_state.processed_files:
        st.subheader("Processed Files")
        for file in st.session_state.processed_files:
            st.write(f"- {file}")

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
            with st.spinner("Generating response......⌛"):
                start = time.time()
                retrieval_chain = setup_retrieval_chain()
                response = retrieval_chain.invoke({'input': user_question})
                answer = response['answer']
                response_time = time.time() - start
                with st.chat_message("assistant"):
                    st.markdown(answer)
                    st.info(f"Response Time {response_time:.2f} seconds")
                    print(f"Response time  for {user_question}: {response_time:.2f} seconds")
                st.session_state.messages.append({"role": "assistant", "content": answer})
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")
            st.error("Please try rephrasing your question or checking if the documents are properly processed.")

# Add a reset button at the bottom of the sidebar
with st.sidebar:
    if st.button("Reset All"):
        st.session_state.vectors = None
        st.session_state.messages = []
        st.session_state.processed_files = []

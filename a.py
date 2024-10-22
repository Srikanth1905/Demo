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
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import (
    PyMuPDFLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
    Docx2txtLoader
)
import glob
import warnings
import tempfile

# Configuration and Setup
load_dotenv()
st.set_page_config(page_title="Local Directory RAG Assistant", page_icon="🤖", layout="wide")
st.title("Local Directory RAG Assistant 🤖")

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", module="tensorflow")
warnings.filterwarnings("ignore", module="langchain_community")
warnings.filterwarnings("ignore", module="langchain")
warnings.filterwarnings("ignore", module="langchain_ollama")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Define the local documents directory path
LOCAL_DOCS_DIR = "./documents"  # Change this to your desired local directory

# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = []

@st.cache_resource
def initialize_llm():
    return OllamaLLM(
        model=os.getenv("OLLAMA_MODEL"),
        temperature=0.1
    )

@st.cache_resource
def initialize_embeddings():
    return HuggingFaceEmbeddings(
        model_name='sentence-transformers/all-MiniLM-L6-v2',
        model_kwargs={'device': 'cpu'}
    )

def get_file_loader(file_path):
    """Get appropriate loader based on file extension"""
    ext = os.path.splitext(file_path)[1].lower()
    loaders = {
        ".pdf": PyMuPDFLoader,
        ".docx": Docx2txtLoader,
        ".txt": TextLoader
    }
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

def process_documents(documents):
    """Process and split documents"""
    if not documents:
        return []
        
    with st.spinner("Processing documents..."):
        avg_doc_length = sum(len(doc.page_content) for doc in documents) / len(documents)
        chunk_size = min(max(500, int(avg_doc_length / 4)), 1000)
        
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=50,
            length_function=len,
            add_start_index=True,
        )

        splits = text_splitter.split_documents(documents)
        st.sidebar.write(f"Created {len(splits)} chunks from {len(documents)} documents")
        
        return splits

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
        2. **Supporting Details**: Expand the answer with details and evidence drawn from the context.
        3. **Limitations**: Mention any uncertainties or limitations if the context is incomplete or ambiguous.
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
    uploaded_files = st.file_uploader(
        "Upload your documents",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True
    )
    
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
                    embeddings = initialize_embeddings()
                    st.session_state.vectors = FAISS.from_documents(splits, embeddings)
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
                retrieval_chain = setup_retrieval_chain()
                response = retrieval_chain.invoke({
                    'input': user_question
                })
                answer = response['answer']
                
                with st.chat_message("assistant"):
                    st.markdown(answer)
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
        

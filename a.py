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
    TextLoader
)
import glob
import warnings
# Configuration and Setup
load_dotenv()
st.set_page_config(page_title="Local Directory RAG Assistant", page_icon="🤖", layout="wide")
st.title("Local Directory RAG Assistant 🤖")

warnings.filterwarnings("ignore",category=UserWarning)
warnings.filterwarnings("ignore",category=DeprecationWarning)
warnings.filterwarnings("ignore",module="tensorflow")
warnings.filterwarnings("ignore",module="langchain_community")
warnings.filterwarnings("ignore",module="langchain")
warnings.filterwarnings("ignore",module="langchain_ollama")

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Initialize session state
if "vectors" not in st.session_state:
    st.session_state.vectors = None
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processed_directory" not in st.session_state:
    st.session_state.processed_directory = None

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

def load_documents(directory_path):
    """Load documents using LangChain's DirectoryLoader"""
    loaders = {
        ".pdf": (PyMuPDFLoader, {}),
        ".docx": (UnstructuredWordDocumentLoader, {}),
        ".txt": (TextLoader, {"encoding": "utf8"})
    }
    
    documents = []
    with st.spinner("Loading documents..."):
        for ext in loaders:
            loader_class, loader_args = loaders[ext]
            # Create glob pattern for current extension
            pattern = os.path.join(directory_path, f"*{ext}")
            
            # Use DirectoryLoader with the specific loader for each file type
            if glob.glob(pattern):  # Only create loader if files exist
                loader = DirectoryLoader(
                    directory_path,
                    glob=f"**/*{ext}",  # Include subdirectories
                    loader_cls=loader_class,
                    loader_kwargs=loader_args,
                    recursive=True,
                    show_progress=True,
                    use_multithreading=True
                )
                try:
                    ext_docs = loader.load()
                    st.write(f"Loaded {len(ext_docs)} {ext} files")
                    documents.extend(ext_docs)
                except Exception as e:
                    st.error(f"Error loading {ext} files: {str(e)}")
    
    return documents

def process_documents(documents):
    """Process and split documents"""
    with st.spinner("Processing documents..."):
        # Calculate average document length for optimal chunk size
        if documents:
            avg_doc_length = sum(len(doc.page_content) for doc in documents) / len(documents)
            chunk_size = min(max(500, int(avg_doc_length / 4)), 1000)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size,chunk_overlap=50,length_function=len,add_start_index=True,)

            splits = text_splitter.split_documents(documents)
            st.write(f"Created {len(splits)} chunks from {len(documents)} documents")
            
            return splits
    return []

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

# Streamlit UI
col1, col2 = st.columns([3, 1])

with col1:
    # Chat interface
    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

with col2:
    # Directory input
    directory_path = st.text_input("Enter the path to your documents directory:", "")
    process_button = st.button("Process Directory")
    
    if directory_path and process_button:
        if not os.path.exists(directory_path):
            st.error("Directory not found!")
        elif directory_path != st.session_state.processed_directory:
            try:
                # Load documents using DirectoryLoader
                documents = load_documents(directory_path)
                
                if documents:
                    # Process and split documents
                    splits = process_documents(documents)
                    
                    if splits:
                        # Create vector store
                        with st.spinner("Creating vector store..."):
                            embeddings = initialize_embeddings()
                            st.session_state.vectors = FAISS.from_documents(splits, embeddings)
                            st.session_state.processed_directory = directory_path
                            st.success("Documents processed successfully! Ready for questions.")
                else:
                    st.warning("No supported documents found in the directory.")
            except Exception as e:
                st.error(f"Error processing directory: {str(e)}")
        else:
            st.info("Directory already processed!")

    # Display processing status
    if st.session_state.processed_directory:
        st.write("Currently processed directory:", st.session_state.processed_directory)

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
            with st.spinner("Generating  response......⌛"):
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
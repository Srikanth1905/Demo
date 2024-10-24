import os
import time
import streamlit as st
from dotenv import load_dotenv
from langchain_ollama import OllamaLLM
from langchain_community.document_loaders import DirectoryLoader, TextLoader, UnstructuredWordDocumentLoader, UnstructuredMarkdownLoader, PyMuPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# Configuration and Setup
load_dotenv()
st.set_page_config(page_title="LLM Assistant", page_icon="📚", layout="centered")

# Constants
LOCAL_DOCS_DIR = "./cyberdata"

def create_vector_store():
    if "vectors" not in st.session_state:
        with st.spinner("Initializing embeddings..."):
            st.session_state.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-l6-v2")
        all_docs = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Define loaders
        loaders = [("Text files", "**/*.txt", TextLoader, {'encoding': 'utf-8'}),
            ("Word documents", "**/*.docx", UnstructuredWordDocumentLoader, {}),
            ("Markdown files", "**/*.md", UnstructuredMarkdownLoader, {}),
            ("PDF files", "**/*.pdf", PyMuPDFLoader, {})
        ]
        
        # Load documents with progress tracking
        for idx, (file_type, glob_pattern, loader_cls, loader_kwargs) in enumerate(loaders):
            progress = (idx / len(loaders)) * 0.5  # First 50% for loading
            status_text.text(f"Loading {file_type}...")
            progress_bar.progress(progress)
            
            loader = DirectoryLoader(LOCAL_DOCS_DIR,glob=glob_pattern,loader_cls=loader_cls,use_multithreading=True,loader_kwargs=loader_kwargs)
            docs = loader.load()
            all_docs.extend(docs)
        
        # Process documents
        status_text.text("Processing documents...")
        progress_bar.progress(0.6)
        
        st.session_state.docs = all_docs
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=6000,
            chunk_overlap=1000
        )
        
        status_text.text("Splitting documents...")
        progress_bar.progress(0.7)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:100]
        )
        
        status_text.text("Creating vector store...")
        progress_bar.progress(0.8)
        st.session_state.vectors = FAISS.from_documents(
            st.session_state.final_documents,
            st.session_state.embeddings
        )
        
        progress_bar.progress(1.0)
        status_text.text("Vector store created successfully!")
        st.success(f"Processed {len(all_docs)} documents into {len(st.session_state.final_documents)} chunks")

def create_rag_prompt():
    return ChatPromptTemplate.from_template("""Answer strictly based on the provided context. Do not include any external knowledge.

Context: {context}
Question: {input}

Instructions:
- If the question can be answered from the context, provide a clear, direct answer using only information from the documents
- If the question cannot be answered from the context, respond only with: "This question cannot be answered from the provided documents."
- Do not speculate or add information beyond what's in the context
- If quoting from the documents, use quotation marks

Answer:
""")

@st.cache_resource
def setup_retrieval_chain():
    llm = OllamaLLM(model=os.getenv('OLLAMA_MODEL'))
    prompt = create_rag_prompt()
    document_chain = create_stuff_documents_chain(llm,prompt,document_prompt=ChatPromptTemplate.from_template("{page_content}"),document_separator="\n\n")
    retriever = st.session_state.vectors.as_retriever(search_type="similarity",search_kwargs={"k": 3})
    return create_retrieval_chain(retriever, document_chain)

# UI Layout
st.title("📚 LLM RAG Assistant")
st.write("Ask me.......")

# Initialize vector store
if st.button("Process Documents", type="primary"):
    create_vector_store()

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input
if prompt := st.chat_input("Ask about your documents...", key="chat_input"):
    if "vectors" not in st.session_state:
        st.warning("⚠️ Please process documents first!")
    else:
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        try:
            start_time = time.time()
            with st.spinner("Thinking..."):
                retrieval_chain = setup_retrieval_chain()
                response = retrieval_chain.invoke({'input': prompt})
                
                # Calculate response time
                response_time = time.time() - start_time
                
                # Add assistant message with response time
                st.session_state.messages.append({"role": "assistant","content": f"{response['answer']}\n\n_Response time: {response_time:.2f} seconds_"})
                st.rerun()
                
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

# Reset button
if st.button("Reset", type="secondary", help="Clear all processed documents and chat history"):
    for key in ['vectors', 'docs', 'text_splitter', 'final_documents', 'embeddings', 'messages']:
        if key in st.session_state:
            del st.session_state[key]
    st.success("🧹 All data has been reset!")

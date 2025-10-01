import logging
import os
import time
import streamlit as st
from PyPDF2 import PdfReader

from src.constants import OPENSEARCH_INDEX, TEXT_CHUNK_SIZE, UPLOAD_DIR
from src.embeddings import generate_embeddings, get_embedding_model
from src.ingestion import bulk_index_documents, create_index, delete_documents_by_document_name
from src.opensearch import get_opensearch_client
from src.utils import chunk_text, setup_logging, secure_file_path, validate_file_type

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Document Upload", page_icon="📄")

def apply_custom_css():
    """Apply custom styling."""
    st.markdown("""
    <style>
    .upload-area { border: 2px dashed #118ab2; padding: 2rem; border-radius: 10px; text-align: center; }
    .doc-item { background: #f8f9fa; padding: 1rem; border-radius: 5px; margin: 0.5rem 0; }
    .delete-btn { background-color: #dc3545 !important; }
    </style>
    """, unsafe_allow_html=True)

def setup_sidebar():
    """Configure sidebar."""
    st.sidebar.title("📄 Document Manager")
    
    # Status
    if st.session_state.get("services_ready"):
        st.sidebar.success("✅ Services Ready")
    else:
        st.sidebar.warning("⏳ Loading...")
    
    # Logo
    logo_path = os.path.join("images", "jamwithai_logo.png")
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=200)
    
    st.sidebar.markdown("---")
    st.sidebar.info("Upload PDFs to enable document search in the chatbot.")

def initialize_services():
    """Initialize required services."""
    if "services_ready" not in st.session_state:
        with st.spinner("Initializing document services..."):
            try:
                # Initialize OpenSearch
                client = get_opensearch_client()
                if client:
                    create_index(client)
                    st.success("✅ OpenSearch connected")
                else:
                    st.error("❌ OpenSearch not available")
                    return False
                
                # Load embedding model
                model = get_embedding_model()
                if model:
                    st.success("✅ Embedding model loaded")
                else:
                    st.error("❌ Embedding model failed to load")
                    return False
                
                st.session_state["services_ready"] = True
                return True
                
            except Exception as e:
                logger.error(f"Service initialization failed: {e}")
                st.error(f"❌ Service initialization failed: {e}")
                return False
    return True

def get_indexed_documents():
    """Get list of documents from OpenSearch index."""
    client = get_opensearch_client()
    if not client:
        return []
    
    try:
        # Use aggregation with proper field mapping
        query = {
            "size": 0,
            "aggs": {
                "unique_docs": {
                    "terms": {
                        "field": "document_name.keyword",
                        "size": 10000
                    }
                }
            }
        }
        response = client.search(index=OPENSEARCH_INDEX, body=query)
        
        # Fallback to simple search if aggregation fails
        if "aggregations" not in response:
            query = {"size": 1000, "_source": ["document_name"]}
            response = client.search(index=OPENSEARCH_INDEX, body=query)
            docs = set()
            for hit in response["hits"]["hits"]:
                doc_name = hit["_source"].get("document_name")
                if doc_name:
                    docs.add(doc_name)
            return list(docs)
        
        buckets = response["aggregations"]["unique_docs"]["buckets"]
        return [bucket["key"] for bucket in buckets]
    except Exception as e:
        logger.error(f"Failed to get indexed documents: {e}")
        return []

def save_uploaded_file(uploaded_file) -> str:
    """Save uploaded file securely."""
    try:
        # Validate file type
        if not validate_file_type(uploaded_file.name, ['.pdf']):
            raise ValueError("Only PDF files are allowed")
        
        # Create secure file path
        file_path = secure_file_path(UPLOAD_DIR, uploaded_file.name)
        
        # Save file
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        logger.info(f"File saved securely: {uploaded_file.name}")
        return file_path
        
    except Exception as e:
        logger.error(f"Failed to save file {uploaded_file.name}: {e}")
        raise

def process_document(uploaded_file, existing_docs):
    """Process and index a single document."""
    try:
        # Check if already exists
        if uploaded_file.name in existing_docs:
            st.warning(f"📄 {uploaded_file.name} already exists")
            return False
        
        # Save file securely
        file_path = save_uploaded_file(uploaded_file)
        
        # Extract text
        with st.spinner(f"Processing {uploaded_file.name}..."):
            reader = PdfReader(file_path)
            text = "".join([page.extract_text() for page in reader.pages])
            
            if not text.strip():
                st.warning(f"No text found in {uploaded_file.name}")
                return False
            
            # Create chunks and embeddings
            chunks = chunk_text(text, words_per_chunk=TEXT_CHUNK_SIZE, overlap=100)
            embeddings = generate_embeddings(chunks)
            
            if embeddings is None or len(embeddings) == 0:
                st.error(f"Failed to generate embeddings for {uploaded_file.name}")
                return False
            
            # Prepare documents for indexing
            documents_to_index = [
                {
                    "doc_id": f"{uploaded_file.name}_{i}",
                    "text": chunk,
                    "embedding": embedding,
                    "document_name": uploaded_file.name,
                }
                for i, (chunk, embedding) in enumerate(zip(chunks, embeddings))
            ]
            
            # Index documents
            success, errors = bulk_index_documents(documents_to_index)
            
            if success > 0:
                st.success(f"✅ {uploaded_file.name} processed ({len(chunks)} chunks)")
                return True
            else:
                st.error(f"❌ Failed to index {uploaded_file.name}")
                return False
                
    except Exception as e:
        st.error(f"❌ Error processing {uploaded_file.name}: {e}")
        logger.error(f"Document processing error: {e}")
        return False

def handle_file_upload():
    """Handle file upload interface."""
    st.markdown("### Upload Documents")
    
    # Get existing documents
    existing_docs = get_indexed_documents()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type="pdf",
        accept_multiple_files=True,
        help="Upload PDF documents to enable AI search"
    )
    
    if uploaded_files:
        success_count = 0
        
        for uploaded_file in uploaded_files:
            if process_document(uploaded_file, existing_docs):
                success_count += 1
                existing_docs.append(uploaded_file.name)
        
        if success_count > 0:
            st.success(f"🎉 Successfully processed {success_count} document(s)!")

def display_documents():
    """Display and manage existing documents."""
    documents = get_indexed_documents()
    
    if not documents:
        st.info("📝 No documents uploaded yet. Upload some PDFs to get started!")
        return
    
    st.markdown("### Manage Documents")
    
    with st.expander(f"📚 {len(documents)} Documents", expanded=True):
        for i, doc_name in enumerate(documents, 1):
            col1, col2 = st.columns([4, 1])
            
            with col1:
                st.markdown(f"**{i}.** {doc_name}")
            
            with col2:
                if st.button("🗑️", key=f"delete_{doc_name}", help=f"Delete {doc_name}"):
                    with st.spinner(f"Deleting {doc_name}..."):
                        if delete_documents_by_document_name(doc_name):
                            # Also try to delete local file
                            try:
                                file_path = secure_file_path(UPLOAD_DIR, doc_name)
                                if os.path.exists(file_path):
                                    os.remove(file_path)
                            except (ValueError, OSError) as e:
                                logger.warning(f"Could not delete local file {doc_name}: {e}")
                            
                            st.success(f"✅ Deleted {doc_name}")
                            time.sleep(1)
                            st.rerun()
                        else:
                            st.error(f"❌ Failed to delete {doc_name}")

def main():
    """Main upload page function."""
    apply_custom_css()
    setup_sidebar()
    
    st.title("📄 Document Upload")
    st.markdown("Upload PDF documents to enable AI-powered search and chat.")
    
    # Initialize services
    if not initialize_services():
        st.error("Please ensure OpenSearch is running and try again.")
        
        with st.expander("Setup Help"):
            st.markdown("""
            **Requirements:**
            - OpenSearch running on localhost:9200
            - Internet connection for embedding model
            
            **Start OpenSearch:**
            ```bash
            # Using Docker
            docker run -p 9200:9200 -e "discovery.type=single-node" opensearchproject/opensearch:latest
            ```
            """)
        return
    
    # Main functionality
    handle_file_upload()
    st.markdown("---")
    display_documents()

if __name__ == "__main__":
    main()

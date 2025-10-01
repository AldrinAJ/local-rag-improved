import logging
import os
import streamlit as st

# Cache logo path at module level
LOGO_PATH = os.path.join(os.path.dirname(__file__), "images", "logo.png")
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="AI Document Assistant", 
    page_icon="🤖",
    layout="wide"
)

def apply_custom_css():
    """Apply modern styling."""
    st.markdown("""
    <style>
    .main-header { color: #006d77; text-align: center; }
    .feature-box { 
        background: #f0f8ff; 
        padding: 1rem; 
        border-radius: 10px; 
        margin: 0.5rem 0;
    }
    .stButton button { 
        background-color: #118ab2; 
        color: white; 
        border-radius: 5px;
    }
    </style>
    """, unsafe_allow_html=True)

def display_sidebar():
    """Display sidebar content."""
    st.sidebar.markdown("## 🤖 OpenSearch AI Assistant")
    
    logo_path = LOGO_PATH
    try:
        if os.path.exists(logo_path):
            st.sidebar.image(logo_path, width=200)
        else:
            st.sidebar.info("Logo placeholder")
    except (OSError, PermissionError):
        st.sidebar.info("Logo placeholder")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("© 2025 OpenSearch AI Document Assistant")

def main():
    """Main welcome page."""
    apply_custom_css()
    display_sidebar()
    
    st.markdown('<h1 class="main-header">🤖 OpenSearch AI Document Assistant</h1>', unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to your intelligent document processing assistant! 
    
    **Features:**
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="feature-box">
        <h3>🤖 Smart Chatbot</h3>
        <p>Interact with OpenSearch AI using language models with RAG capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-box">
        <h3>📄 Document Upload</h3>
        <p>Upload and process PDFs with hybrid search using OpenSearch.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("### Get Started")
    st.info("👈 Choose a page from the sidebar to begin!")
    
    logger.info("Welcome page displayed")

if __name__ == "__main__":
    main()
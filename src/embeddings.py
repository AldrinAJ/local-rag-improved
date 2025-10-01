import logging
from typing import List, Optional
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer

from src.constants import EMBEDDING_MODEL_PATH
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def get_embedding_model() -> Optional[SentenceTransformer]:
    """Load and cache embedding model with error handling."""
    try:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL_PATH}")
        model = SentenceTransformer(EMBEDDING_MODEL_PATH)
        logger.info("Embedding model loaded successfully")
        return model
    except (OSError, ValueError, RuntimeError) as e:
        logger.error(f"Failed to load embedding model: {e}")
        st.error("Failed to load embedding model. Please check your internet connection.")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading embedding model: {e}")
        st.error("Unexpected error loading embedding model.")
        return None

def generate_embeddings(chunks: List[str]) -> List[np.ndarray]:
    """Generate embeddings efficiently using batch processing."""
    model = get_embedding_model()
    if not model or not chunks:
        return []
    
    try:
        # Batch processing for better performance
        embeddings = model.encode(chunks, batch_size=32, show_progress_bar=False)
        
        logger.info(f"Generated {len(embeddings)} embeddings using batch processing")
        return embeddings
    except Exception as e:
        logger.error(f"Embedding generation failed: {e}")
        return []

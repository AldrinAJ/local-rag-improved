import logging
import os
import re
from pathlib import Path
from typing import List
from werkzeug.utils import secure_filename

from src.constants import LOG_FILE_PATH, UPLOAD_DIR

def setup_logging() -> None:
    """Setup logging with both file and console output."""
    log_dir = os.path.dirname(LOG_FILE_PATH)
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
    
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(LOG_FILE_PATH),
                logging.StreamHandler()  # Console output for development
            ],
            force=True
        )

def clean_text(text: str) -> str:
    """Clean text with detailed logging."""
    original_length = len(text)
    
    # Remove extra whitespace and normalize
    text = re.sub(r'\s+', ' ', text.strip())
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', '', text)
    
    cleaned_length = len(text)
    logging.getLogger(__name__).info(
        f"Text cleaned: {original_length} -> {cleaned_length} chars"
    )
    return text

def chunk_text(text: str, words_per_chunk: int = 300, overlap: int = 50) -> List[str]:
    """Chunk text with proper validation and efficient processing."""
    if words_per_chunk <= 0:
        raise ValueError("words_per_chunk must be positive")
    if overlap >= words_per_chunk:
        raise ValueError("overlap must be less than words_per_chunk")
    if not text.strip():
        return []
    
    # Efficient tokenization
    words = text.split()
    if not words:
        return []
    
    chunks = []
    start = 0
    
    while start < len(words):
        # Calculate end position
        end = min(start + words_per_chunk, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        
        # Move start position with overlap
        if end >= len(words):
            break
        start = end - overlap
    
    logging.getLogger(__name__).info(f"Created {len(chunks)} chunks from text")
    return chunks

def secure_file_path(base_dir: str, filename: str) -> str:
    """Create secure file path preventing directory traversal."""
    # Secure the filename
    safe_name = secure_filename(filename)
    if not safe_name:
        raise ValueError("Invalid filename")
    
    # Ensure base directory exists
    upload_path = Path(base_dir)
    upload_path.mkdir(mode=0o755, exist_ok=True)
    
    # Create full path and verify it's within upload directory
    full_path = upload_path / safe_name
    
    # Resolve paths to prevent traversal
    try:
        full_path = full_path.resolve()
        upload_path = upload_path.resolve()
        
        # Ensure the resolved path is within the upload directory
        try:
            full_path.relative_to(upload_path)
        except ValueError:
            raise ValueError("Path traversal attempt detected")
            
        return str(full_path)
    except (OSError, ValueError) as e:
        raise ValueError(f"Invalid file path: {e}")

def validate_file_type(filename: str, allowed_extensions: List[str] = None) -> bool:
    """Validate file type for security."""
    if allowed_extensions is None:
        allowed_extensions = ['.pdf', '.txt', '.docx', '.png', '.jpg', '.jpeg']
    
    file_ext = Path(filename).suffix.lower()
    return file_ext in allowed_extensions

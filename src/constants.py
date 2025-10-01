EMBEDDING_MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"
ASYMMETRIC_EMBEDDING = False  # Fixed spelling
EMBEDDING_DIMENSION = 768
TEXT_CHUNK_SIZE = 300  # Maximum characters per chunk for document processing

OLLAMA_MODEL_NAME = "qwen3:4b"
OPENAI_MODEL_NAME = "gpt-4o-mini"  # Better cost/performance

# System settings
LOG_FILE_PATH = "logs/app.log"
OPENSEARCH_HOST = "localhost"
OPENSEARCH_PORT = 9200
OPENSEARCH_INDEX = "documents"  # Default fallback index
UPLOAD_DIR = "uploaded_files"  # Secure upload directory

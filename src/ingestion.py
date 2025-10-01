import json
import logging
import os
from typing import Any, Dict, List, Tuple, Optional
from opensearchpy import OpenSearch, helpers, RequestError, ConnectionError

from src.constants import ASYMMETRIC_EMBEDDING, EMBEDDING_DIMENSION, OPENSEARCH_INDEX
from src.opensearch import get_opensearch_client
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

def load_index_config() -> Optional[Dict[str, Any]]:
    """Load index configuration with error handling."""
    try:
        config_path = os.path.join("src", "index_config.json")
        with open(config_path, "r") as f:
            config = json.load(f)
        
        # Safely update embedding dimension
        if ("mappings" in config and 
            "properties" in config["mappings"] and 
            "embedding" in config["mappings"]["properties"]):
            config["mappings"]["properties"]["embedding"]["dimension"] = EMBEDDING_DIMENSION
        logger.info("Index configuration loaded successfully")
        return config
    except FileNotFoundError:
        logger.error("Index configuration file not found")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file: {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        return None

def create_index(client: Optional[OpenSearch]) -> bool:
    """Create index with proper error handling."""
    if not client:
        return False
        
    config = load_index_config()
    if not config:
        return False
    
    try:
        if not client.indices.exists(index=OPENSEARCH_INDEX):
            client.indices.create(index=OPENSEARCH_INDEX, body=config)
            logger.info(f"Created index {OPENSEARCH_INDEX}")
        else:
            logger.info(f"Index {OPENSEARCH_INDEX} already exists")
        return True
    except RequestError as e:
        logger.error(f"Failed to create index: {e}")
        return False
    except Exception as e:
        logger.error(f"Index creation failed: {e}")
        return False

def bulk_index_documents(documents: List[Dict[str, Any]]) -> Tuple[int, List[Any]]:
    """Index documents with improved error handling."""
    client = get_opensearch_client()
    if not client or not documents:
        return 0, ["No client or documents available"]
    
    actions = []
    for doc in documents:
        try:
            # Validate required fields
            required_fields = ["doc_id", "text", "embedding", "document_name"]
            for field in required_fields:
                if field not in doc:
                    raise KeyError(f"Missing required field '{field}'")
            
            # Handle different embedding types
            embedding = doc["embedding"]
            if hasattr(embedding, 'tolist'):
                embedding_list = embedding.tolist()
            elif isinstance(embedding, list):
                embedding_list = embedding
            else:
                embedding_list = list(embedding)
            text = doc["text"]
            
            # Apply asymmetric prefix if needed
            if ASYMMETRIC_EMBEDDING:
                text = f"passage: {text}"
            
            action = {
                "_index": OPENSEARCH_INDEX,
                "_id": doc["doc_id"],
                "_source": {
                    "text": text,
                    "embedding": embedding_list,
                    "document_name": doc["document_name"],
                },
            }
            actions.append(action)
        except Exception as e:
            logger.error(f"Failed to prepare document {doc.get('doc_id', 'unknown')}: {e}")
    
    if not actions:
        return 0, ["No valid documents to index"]
    
    try:
        success, errors = helpers.bulk(client, actions)
        logger.info(f"Indexed {success} documents with {len(errors)} errors")
        return success, errors
    except Exception as e:
        logger.error(f"Bulk indexing failed: {e}")
        return 0, [str(e)]

def delete_documents_by_document_name(document_name: str) -> bool:
    """Delete documents with improved error handling."""
    client = get_opensearch_client()
    if not client:
        return False
    
    try:
        # Use term query for exact matching
        query = {"query": {"term": {"document_name": document_name}}}
        response = client.delete_by_query(index=OPENSEARCH_INDEX, body=query)
        
        deleted_count = response.get("deleted", 0)
        logger.info(f"Deleted {deleted_count} documents for '{document_name}'")
        return True
    except Exception as e:
        logger.error(f"Failed to delete documents: {e}")
        return False

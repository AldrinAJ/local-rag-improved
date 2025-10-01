import logging
from typing import Any, Dict, List, Optional
import streamlit as st
from opensearchpy import OpenSearch, ConnectionError, RequestError

from src.constants import OPENSEARCH_HOST, OPENSEARCH_INDEX, OPENSEARCH_PORT
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

@st.cache_resource(show_spinner=False)
def get_opensearch_client() -> Optional[OpenSearch]:
    """Initialize and cache OpenSearch client with error handling."""
    try:
        client = OpenSearch(
            hosts=[{"host": OPENSEARCH_HOST, "port": OPENSEARCH_PORT}],
            http_compress=True,
            timeout=30,
            max_retries=3,
            retry_on_timeout=True,
        )
        # Test connection
        client.info()
        logger.info("OpenSearch client initialized successfully")
        return client
    except ConnectionError as e:
        logger.error(f"Failed to connect to OpenSearch: {e}")
        st.error("Cannot connect to OpenSearch. Please check if it's running.")
        return None
    except Exception as e:
        logger.error(f"OpenSearch client initialization failed: {e}")
        st.error(f"OpenSearch initialization failed: {str(e)}")
        return None

# Cache client at module level
_opensearch_client = get_opensearch_client()

def get_available_indices() -> List[str]:
    """Get list of available OpenSearch indices."""
    client = _opensearch_client
    if not client:
        return []
    
    try:
        indices = client.indices.get_alias("*")
        # Filter out system indices
        user_indices = [name for name in indices.keys() if not name.startswith(".")]
        logger.info(f"Found {len(user_indices)} user indices")
        return sorted(user_indices)
    except Exception as e:
        logger.error(f"Failed to get indices: {e}")
        return []

def get_index_fields(index_name: str) -> Dict[str, List[str]]:
    """Get text and vector fields from index mapping."""
    client = _opensearch_client
    if not client:
        return {"text_fields": [], "vector_fields": []}
    
    try:
        mapping = client.indices.get_mapping(index=index_name)
        properties = mapping[index_name]["mappings"].get("properties", {})
        
        text_fields = []
        vector_fields = []
        
        for field_name, field_config in properties.items():
            field_type = field_config.get("type", "")
            if field_type == "text":
                text_fields.append(field_name)
            elif field_type == "knn_vector":
                vector_fields.append(field_name)
            # Check if field has embedding data but wrong type
            elif field_name == "embedding":
                # Check if documents actually have embedding arrays
                try:
                    sample_query = {"size": 1, "_source": [field_name]}
                    sample_response = client.search(index=index_name, body=sample_query)
                    if sample_response["hits"]["hits"]:
                        embedding_value = sample_response["hits"]["hits"][0]["_source"].get(field_name)
                        if isinstance(embedding_value, list) and len(embedding_value) > 0:
                            logger.warning(f"Field '{field_name}' contains embeddings but is not knn_vector type")
                            # Don't add to vector_fields since it's not properly typed
                except Exception:
                    pass
        
        logger.info(f"Index {index_name}: {len(text_fields)} text fields, {len(vector_fields)} vector fields")
        return {"text_fields": text_fields, "vector_fields": vector_fields}
    except Exception as e:
        logger.error(f"Failed to get fields for index {index_name}: {e}")
        return {"text_fields": [], "vector_fields": []}

def simple_text_search(query_text: str, index_name: str, text_field: str = "text", top_k: int = 5) -> List[Dict[str, Any]]:
    """Simple text search for debugging."""
    client = _opensearch_client
    if not client:
        return []
    
    query_body = {
        "query": {"match": {text_field: {"query": query_text}}},
        "size": top_k
    }
    
    try:
        response = client.search(index=index_name, body=query_body)
        logger.info(f"Simple search found {len(response['hits']['hits'])} results")
        return response["hits"]["hits"]
    except Exception as e:
        logger.error(f"Simple search failed: {e}")
        return []

def hybrid_search(
    query_text: str, query_embedding: List[float], top_k: int = 5, index_name: str = None, text_field: str = "text", vector_field: str = "embedding"
) -> List[Dict[str, Any]]:
    """Perform hybrid search with error handling."""
    client = _opensearch_client
    if not client:
        return []

    # For documents without embeddings or improper vector field type, use simple text search
    if not query_embedding or not vector_field:
        logger.info(f"Using simple text search on field '{text_field}' (no embeddings or vector field)")
        return simple_text_search(query_text, index_name or OPENSEARCH_INDEX, text_field, top_k)
    
    # Check if vector field is properly configured as knn_vector
    try:
        client = _opensearch_client
        if client:
            mapping = client.indices.get_mapping(index=index_name or OPENSEARCH_INDEX)
            index_to_check = index_name or OPENSEARCH_INDEX
            properties = mapping[index_to_check]["mappings"].get("properties", {})
            vector_field_type = properties.get(vector_field, {}).get("type", "")
            
            if vector_field_type != "knn_vector":
                logger.warning(f"Vector field '{vector_field}' is not knn_vector type, using text search only")
                return simple_text_search(query_text, index_name or OPENSEARCH_INDEX, text_field, top_k)
    except Exception as e:
        logger.warning(f"Could not check vector field type: {e}, falling back to text search")
        return simple_text_search(query_text, index_name or OPENSEARCH_INDEX, text_field, top_k)
    
    # Build queries based on available fields
    queries = []
    
    # Add text search if text field exists
    if text_field:
        queries.append({"match": {text_field: {"query": query_text}}})
    
    # Add vector search if vector field exists and embedding provided
    if vector_field and query_embedding:
        queries.append({
            "knn": {
                vector_field: {
                    "vector": query_embedding,
                    "k": top_k,
                }
            }
        })
    
    # Fallback to simple text search if no hybrid queries
    if not queries:
        query_body = {
            "query": {"match_all": {}},
            "size": top_k
        }
    elif len(queries) == 1:
        query_body = {
            "_source": {"exclude": [vector_field] if vector_field else []},
            "query": queries[0],
            "size": top_k
        }
    else:
        query_body = {
            "_source": {"exclude": [vector_field] if vector_field else []},
            "query": {"hybrid": {"queries": queries}},
            "size": top_k
        }

    try:
        index_to_use = index_name or OPENSEARCH_INDEX
        
        # Use search pipeline only for hybrid queries
        if len(queries) > 1:
            response = client.search(
                index=index_to_use, 
                body=query_body, 
                search_pipeline="nlp-search-pipeline"
            )
        else:
            # Simple search without pipeline
            response = client.search(
                index=index_to_use, 
                body=query_body
            )
        
        logger.info(f"Search completed: {len(response['hits']['hits'])} results for query: '{query_text}'")
        logger.info(f"Query body: {query_body}")
        return response["hits"]["hits"]
    except RequestError as e:
        logger.error(f"Search request failed: {e}")
        # Try without pipeline as fallback
        try:
            response = client.search(index=index_to_use, body=query_body)
            logger.info(f"Fallback search completed: {len(response['hits']['hits'])} results")
            return response["hits"]["hits"]
        except Exception as fallback_e:
            logger.error(f"Fallback search also failed: {fallback_e}")
            return []
    except Exception as e:
        logger.error(f"Search operation failed: {e}")
        return []

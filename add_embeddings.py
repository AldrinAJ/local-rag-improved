#!/usr/bin/env python3
"""Add embeddings to existing documents in OpenSearch."""

import logging
from src.constants import OPENSEARCH_INDEX
from src.opensearch import get_opensearch_client
from src.embeddings import get_embedding_model
from opensearchpy import helpers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def add_embeddings_to_existing_docs(index_name=None, text_field="text"):
    """Add embeddings to documents that don't have them."""
    client = get_opensearch_client()
    model = get_embedding_model()
    
    if not client or not model:
        logger.error("Failed to initialize client or model")
        return
    
    # Use provided index or default
    target_index = index_name or OPENSEARCH_INDEX
    
    # Get documents without embeddings
    query = {
        "query": {"bool": {"must_not": {"exists": {"field": "embedding"}}}},
        "size": 100  # Process in batches
    }
    
    try:
        response = client.search(index=target_index, body=query)
        docs = response["hits"]["hits"]
        
        if not docs:
            logger.info(f"No documents need embeddings in index '{target_index}'")
            return
        
        logger.info(f"Adding embeddings to {len(docs)} documents in index '{target_index}'")
        
        # Prepare updates
        actions = []
        for doc in docs:
            text = doc["_source"].get(text_field, "")
            if text:
                embedding = model.encode(text).tolist()
                
                actions.append({
                    "_op_type": "update",
                    "_index": target_index,
                    "_id": doc["_id"],
                    "doc": {"embedding": embedding}
                })
        
        # Bulk update
        if actions:
            success, errors = helpers.bulk(client, actions)
            logger.info(f"Updated {success} documents with embeddings")
            if errors:
                logger.error(f"Errors: {errors}")
    
    except Exception as e:
        logger.error(f"Failed to add embeddings: {e}")

if __name__ == "__main__":
    import sys
    
    # Usage: python add_embeddings.py [index_name] [text_field]
    index_name = sys.argv[1] if len(sys.argv) > 1 else None
    text_field = sys.argv[2] if len(sys.argv) > 2 else "text"
    
    print(f"Adding embeddings to index: {index_name or 'default'}")
    print(f"Using text field: {text_field}")
    
    add_embeddings_to_existing_docs(index_name, text_field)
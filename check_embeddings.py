#!/usr/bin/env python3
"""Check and add embeddings to documents."""

import logging
from src.opensearch import get_opensearch_client
from src.embeddings import get_embedding_model
from opensearchpy import helpers

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def check_and_add_embeddings(index_name, text_field="message"):
    """Check if embeddings exist and add them if missing."""
    client = get_opensearch_client()
    model = get_embedding_model()
    
    if not client or not model:
        logger.error("Failed to initialize client or model")
        return
    
    # Check a sample document
    sample_query = {"size": 1, "_source": ["embedding", text_field]}
    response = client.search(index=index_name, body=sample_query)
    
    if response["hits"]["hits"]:
        doc = response["hits"]["hits"][0]["_source"]
        if "embedding" in doc:
            print(f"✅ Embeddings found in index '{index_name}'")
            print(f"Embedding length: {len(doc['embedding'])}")
            return
        else:
            print(f"❌ No embeddings found in index '{index_name}'")
    
    # Add embeddings to all documents
    print(f"Adding embeddings to all documents in '{index_name}'...")
    
    # Get all documents
    query = {"query": {"match_all": {}}, "size": 100}
    response = client.search(index=index_name, body=query)
    docs = response["hits"]["hits"]
    
    if not docs:
        print("No documents found")
        return
    
    # Process documents
    actions = []
    for doc in docs:
        text = doc["_source"].get(text_field, "")
        if text:
            embedding = model.encode(text).tolist()
            actions.append({
                "_op_type": "update",
                "_index": index_name,
                "_id": doc["_id"],
                "doc": {"embedding": embedding}
            })
    
    if actions:
        success, errors = helpers.bulk(client, actions)
        print(f"✅ Added embeddings to {success} documents")
        if errors:
            print(f"❌ {len(errors)} errors occurred")
    else:
        print("No documents to process")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python check_embeddings.py <index_name> [text_field]")
        sys.exit(1)
    
    index_name = sys.argv[1]
    text_field = sys.argv[2] if len(sys.argv) > 2 else "message"
    
    print(f"Checking embeddings in index: {index_name}")
    print(f"Using text field: {text_field}")
    
    check_and_add_embeddings(index_name, text_field)
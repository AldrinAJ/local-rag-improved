#!/usr/bin/env python3
"""Fix vector field mapping for existing indices with embeddings."""

import logging
import sys
from src.opensearch import get_opensearch_client
from src.constants import EMBEDDING_DIMENSION

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fix_vector_mapping(index_name, vector_field="embedding", text_field="text"):
    """Create new KNN-enabled index, migrate data, and add embeddings."""
    client = get_opensearch_client()
    if not client:
        logger.error("Failed to get OpenSearch client")
        return False
    
    try:
        # Check if index exists and ensure it's open
        if not client.indices.exists(index=index_name):
            logger.error(f"Index '{index_name}' does not exist")
            return False
        
        # Ensure source index is open
        logger.info(f"Ensuring index '{index_name}' is open")
        try:
            client.indices.open(index=index_name)
        except Exception:
            pass  # Index might already be open
        
        # Get current index mapping and settings
        current_mapping = client.indices.get_mapping(index=index_name)
        current_settings = client.indices.get_settings(index=index_name)
        
        # Create new index name
        new_index_name = f"{index_name}_knn"
        
        # Check if KNN index already exists
        if client.indices.exists(index=new_index_name):
            logger.info(f"KNN index '{new_index_name}' already exists, using it")
            # Use physical index name for operations
            target_index = new_index_name
        else:
            # Prepare new index configuration
            properties = current_mapping[index_name]["mappings"].get("properties", {})
            properties[vector_field] = {
                "type": "knn_vector",
                "dimension": EMBEDDING_DIMENSION
            }
            
            new_index_body = {
                "settings": {
                    "index": {
                        "knn": True,
                        "number_of_shards": current_settings[index_name]["settings"]["index"].get("number_of_shards", "1"),
                        "number_of_replicas": current_settings[index_name]["settings"]["index"].get("number_of_replicas", "0")
                    }
                },
                "mappings": {
                    "properties": properties
                }
            }
            
            # Create new index
            logger.info(f"Creating new KNN-enabled index '{new_index_name}'")
            client.indices.create(index=new_index_name, body=new_index_body)
            
            # Reindex data directly without pre-checking
            logger.info(f"Migrating data from '{index_name}' to '{new_index_name}'")
            reindex_body = {
                "source": {"index": index_name},
                "dest": {"index": new_index_name}
            }
            reindex_result = client.reindex(body=reindex_body, wait_for_completion=True)
            logger.info(f"Reindex result: {reindex_result}")
            
            # Force refresh the new index
            client.indices.refresh(index=new_index_name)
            
            # Wait a moment for the refresh to complete
            import time
            time.sleep(2)
            
            # Verify reindex worked by checking the new index directly
            verify_query = {"query": {"match_all": {}}, "size": 5}
            verify_response = client.search(index=new_index_name, body=verify_query)
            logger.info(f"Verification: {len(verify_response['hits']['hits'])} documents in new index '{new_index_name}'")
            
            if len(verify_response['hits']['hits']) == 0:
                logger.error("Reindex failed - no documents found in new index")
                return False
            
            # Set target to new index BEFORE deleting old one
            target_index = new_index_name
            logger.info(f"Will add embeddings to physical index: {target_index}")
            
            # Now delete old index and create alias
            logger.info(f"Replacing old index with new KNN-enabled index")
            client.indices.delete(index=index_name)
            client.indices.put_alias(index=new_index_name, name=index_name)
        
        # Add embeddings to documents
        logger.info(f"Adding embeddings to documents in index: {target_index}")
        from src.embeddings import get_embedding_model
        from opensearchpy import helpers
        
        model = get_embedding_model()
        if not model:
            logger.error("Failed to get embedding model")
            return False
        
        # Force refresh before searching
        client.indices.refresh(index=target_index)
        
        # Get documents in smaller batches
        query = {"query": {"match_all": {}}, "size": 10}
        
        logger.info(f"Searching for documents in index: {target_index}")
        response = client.search(index=target_index, body=query)
        docs = response["hits"]["hits"]
        
        logger.info(f"Found {len(docs)} documents")
        
        if docs:
            actions = []
            processed = 0
            
            for i, doc in enumerate(docs):
                text = doc["_source"].get(text_field, "")
                logger.info(f"Document {i+1}: text_field='{text_field}', text_length={len(text) if text else 0}")
                
                if text:
                    # Generate embedding
                    logger.info(f"Generating embedding for document {i+1}")
                    embedding = model.encode(text).tolist()
                    logger.info(f"Generated embedding of length {len(embedding)}")
                    
                    # Use update operation to add embedding field
                    actions.append({
                        "_op_type": "update",
                        "_index": target_index,
                        "_id": doc["_id"],
                        "doc": {vector_field: embedding},
                        "doc_as_upsert": True
                    })
                    processed += 1
                else:
                    logger.warning(f"Document {i+1} has no text in field '{text_field}'")
            
            if actions:
                logger.info(f"Bulk updating {processed} documents with embeddings")
                try:
                    success, errors = helpers.bulk(client, actions, chunk_size=5)
                    logger.info(f"Bulk operation completed: {success} successful")
                    
                    if errors:
                        logger.error(f"Bulk errors: {len(errors)}")
                        for error in errors:
                            logger.error(f"Bulk error: {error}")
                    
                    # Verify embeddings were added
                    verify_query = {"size": 1, "_source": [vector_field]}
                    verify_response = client.search(index=target_index, body=verify_query)
                    if verify_response["hits"]["hits"]:
                        verify_doc = verify_response["hits"]["hits"][0]["_source"]
                        if vector_field in verify_doc:
                            logger.info(f"✅ Verification: Embedding field '{vector_field}' found in documents")
                        else:
                            logger.error(f"❌ Verification: Embedding field '{vector_field}' NOT found in documents")
                    
                except Exception as bulk_error:
                    logger.error(f"Bulk operation failed: {bulk_error}")
                    return False
            else:
                logger.warning("No documents with text content found")
        else:
            logger.warning("No documents found in index")
        
        logger.info("Vector mapping and embedding addition completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to fix vector mapping: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python fix_vector_mapping.py <index_name> [vector_field]")
        sys.exit(1)
    
    index_name = sys.argv[1]
    vector_field = sys.argv[2] if len(sys.argv) > 2 else "embedding"
    
    print(f"Fixing vector mapping for index: {index_name}")
    print(f"Vector field: {vector_field}")
    
    success = fix_vector_mapping(index_name, vector_field)
    if success:
        print("✅ Vector mapping fixed! You can now use semantic search.")
    else:
        print("❌ Failed to fix vector mapping.")
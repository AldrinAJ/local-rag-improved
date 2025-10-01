import logging
import os
from typing import Dict, Iterable, List, Optional

import openai
import ollama
import streamlit as st
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.constants import ASYMMETRIC_EMBEDDING, OPENAI_MODEL_NAME, OLLAMA_MODEL_NAME
from src.embeddings import get_embedding_model
from src.opensearch import hybrid_search
from src.utils import setup_logging

setup_logging()
logger = logging.getLogger(__name__)

# Constants
MAX_HISTORY_MESSAGES = 6  # Limit to prevent token overflow

def get_openai_client() -> Optional[OpenAI]:
    """Initialize OpenAI client using API key from .env file."""
    # Force reload environment variables
    load_dotenv(override=True)
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your-openai-api-key-here" or api_key.strip() == "" or not api_key.startswith("sk-"):
        logger.info("OpenAI API key not configured or invalid, will use Ollama")
        return None
    
    try:
        client = OpenAI(api_key=api_key, timeout=30.0, max_retries=2)
        # Test the client with a minimal request to validate the key
        client.models.list()
        logger.info("OpenAI client initialized and validated successfully")
        return client
    except Exception as e:
        logger.info(f"OpenAI not available or invalid key, will use Ollama: {e}")
        return None

# Get client dynamically to avoid caching with placeholder key
_openai_client = None

def run_ollama_streaming(prompt: str, temperature: float) -> Optional[Iterable[str]]:
    """Uses Ollama as fallback when OpenAI is not available."""
    try:
        logger.info("Streaming response from Ollama model.")
        stream = ollama.chat(
            model=OLLAMA_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
            options={"temperature": temperature},
        )
        
        def content_generator():
            for chunk in stream:
                if isinstance(chunk, dict) and "message" in chunk:
                    content = chunk["message"].get("content", "")
                    if content:
                        yield content
        
        return content_generator()
    except Exception as e:
        logger.error(f"Ollama streaming failed: {e}")
        return iter([f"Error: {str(e)}"])

def run_openai_streaming(messages: List[Dict[str, str]], temperature: float, client: OpenAI) -> Optional[Iterable[str]]:
    """Uses OpenAI's API to run the model with streaming enabled."""
    
    try:
        logger.info("Streaming response from OpenAI model.")
        stream = client.chat.completions.create(
            model=OPENAI_MODEL_NAME,
            messages=messages,
            stream=True,
            temperature=temperature,
            max_tokens=1000
        )
        
        def content_generator():
            try:
                for chunk in stream:
                    # Proper bounds checking for choices array
                    if (chunk.choices and len(chunk.choices) > 0 and 
                        chunk.choices[0].delta.content):
                        yield chunk.choices[0].delta.content
            except Exception as e:
                logger.error(f"Error in content generation: {e}")
                yield f"Error: {str(e)}"
        
        return content_generator()
        
    except openai.RateLimitError:
        logger.error("Rate limit exceeded")
        return iter(["Rate limit exceeded. Please try again later."])
    except openai.AuthenticationError:
        logger.error("Authentication failed")
        return iter(["Authentication failed. Check API key."])
    except Exception as e:
        logger.error(f"Unexpected error during streaming: {e}")
        return iter([f"Error: {str(e)}"])

def prompt_template_openai(query: str, context: str, history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Builds the messages with context, conversation history, and user query in OpenAI format."""
    system_message = "You are a knowledgeable chatbot assistant."
    if context:
        system_message += " Use the following context to answer the question."
    
    messages = [{"role": "system", "content": system_message}]
    
    if context:
        messages.append({"role": "system", "content": f"Context:\n{context}"})
    
    # Add conversation history with proper limit to prevent token overflow
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        if isinstance(msg, dict) and msg.get('role') and msg.get('content'):
            messages.append({"role": msg["role"], "content": msg["content"]})
    
    messages.append({"role": "user", "content": query})
    
    logger.info("Messages constructed with context and conversation history.")
    return messages

def prompt_template_ollama(query: str, context: str, history: List[Dict[str, str]]) -> str:
    """Builds the prompt with context, conversation history, and user query for Ollama."""
    prompt = "You are a knowledgeable chatbot assistant. "
    
    if context:
        prompt += f"Use the following context to answer the question.\nContext:\n{context}\n"
    
    if history:
        prompt += "Conversation History:\n"
        for msg in history[-MAX_HISTORY_MESSAGES:]:
            if msg.get('content'):
                role = 'User' if msg.get('role') == 'user' else 'Assistant'
                prompt += f"{role}: {msg['content']}\n"
        prompt += "\n"
    
    prompt += f"User: {query}\nAssistant:"
    return prompt

def generate_response_streaming(
    query: str,
    use_hybrid_search: bool,
    num_results: int,
    temperature: float,
    chat_history: Optional[List[Dict[str, str]]] = None,
    model_choice: str = "auto",
    selected_index: str = None,
    text_field: str = "text",
    vector_field: str = "embedding",
) -> Optional[Iterable[str]]:
    """Generates a chatbot response by performing hybrid search and incorporating conversation history."""
    history = (chat_history or [])[-MAX_HISTORY_MESSAGES:]  # Limit history
    context = ""

    if use_hybrid_search:
        try:
            logger.info("Performing hybrid search.")
            prefixed_query = f"passage: {query}" if ASYMMETRIC_EMBEDDING else query
            embedding_model = get_embedding_model()
            
            if not embedding_model:
                logger.error("Embedding model not available")
                # For search-only mode, return error about embedding model
                # For AI modes, try to continue without search context
                client = get_openai_client()
                if client:
                    messages = prompt_template_openai(query, "", history)
                    return run_openai_streaming(messages, temperature, client)
                else:
                    prompt = prompt_template_ollama(query, "", history)
                    return run_ollama_streaming(prompt, temperature)
            
            query_embedding = embedding_model.encode(prefixed_query).tolist()
            search_results = hybrid_search(query, query_embedding, top_k=num_results, index_name=selected_index, text_field=text_field, vector_field=vector_field)
            logger.info("Hybrid search completed.")

            # Collect text from search results efficiently
            context_parts = []
            for i, result in enumerate(search_results):
                try:
                    text = result.get('_source', {}).get('text', '')[:500]
                    if text:
                        context_parts.append(f"Document {i}:\n{text}")
                except (KeyError, TypeError):
                    logger.warning(f"Invalid search result format at index {i}")
            context = "\n\n".join(context_parts)
        except Exception as e:
            logger.error(f"Hybrid search failed: {e}")

    if model_choice == "openai":
        client = get_openai_client()
        if client:
            messages = prompt_template_openai(query, context, history)
            return run_openai_streaming(messages, temperature, client)
        else:
            error_msg = "OpenAI not available. Please check your API key in .env file"
            logger.error(error_msg)
            return iter([error_msg])
    elif model_choice == "ollama":
        try:
            # Test Ollama availability
            ollama.list()
            prompt = prompt_template_ollama(query, context, history)
            return run_ollama_streaming(prompt, temperature)
        except Exception as e:
            error_msg = f"Ollama not available. Please ensure Ollama is running. Error: {str(e)}"
            logger.error(error_msg)
            return iter([error_msg])
    elif model_choice == "auto":
        client = get_openai_client()
        if client:
            messages = prompt_template_openai(query, context, history)
            return run_openai_streaming(messages, temperature, client)
        else:
            try:
                # Test Ollama availability
                ollama.list()
                prompt = prompt_template_ollama(query, context, history)
                return run_ollama_streaming(prompt, temperature)
            except Exception as e:
                error_msg = f"No models available. OpenAI: API key not configured. Ollama: {str(e)}"
                logger.error(error_msg)
                return iter([error_msg])
    else:
        return iter([f"Unknown model choice: {model_choice}"])

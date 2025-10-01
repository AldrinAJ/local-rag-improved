import logging
import os

import streamlit as st

from src.chat import generate_response_streaming
from src.embeddings import get_embedding_model
from src.ingestion import create_index
from src.opensearch import get_opensearch_client, get_available_indices, get_index_fields, hybrid_search
from opensearchpy import helpers
from src.constants import OPENSEARCH_INDEX, OLLAMA_MODEL_NAME
import ollama
from src.utils import setup_logging

# Initialize logger
setup_logging()  # Configures logging for the application
logger = logging.getLogger(__name__)

# Page configuration is handled by Welcome.py

# Apply custom CSS
st.markdown(
    """
    <style>
    /* Main background and text colors */
    body { background-color: #f0f8ff; color: #002B5B; }
    .sidebar .sidebar-content { background-color: #006d77; color: white; padding: 20px; border-right: 2px solid #003d5c; }
    .sidebar h2, .sidebar h4 { color: white; }
    .block-container { background-color: white; border-radius: 10px; padding: 20px; box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.1); }
    .footer-text { font-size: 1.1rem; font-weight: bold; color: black; text-align: center; margin-top: 10px; }
    .stButton button { background-color: #118ab2; color: white; border-radius: 5px; padding: 10px 20px; font-size: 16px; }
    .stButton button:hover { background-color: #07a6c2; color: white; }
    h1, h2, h3, h4 { color: #006d77; }
    .stChatMessage { background-color: #e0f7fa; color: #006d77; padding: 10px; border-radius: 5px; margin-bottom: 10px; }
    .stChatMessage.user { background-color: #118ab2; color: white; }
    .thinking-text { color: #888888; font-style: italic; opacity: 0.6; }
    </style>
    """,
    unsafe_allow_html=True,
)
logger.info("Custom CSS applied.")


# Main chatbot page rendering function
def render_chatbot_page() -> None:
    # Set up a placeholder at the very top of the main content area
    st.title("OpenSearch AI - Chatbot 🤖")
    model_loading_placeholder = st.empty()

    # Initialize session state variables for chatbot settings
    if "use_hybrid_search" not in st.session_state:
        st.session_state["use_hybrid_search"] = True
    if "num_results" not in st.session_state:
        st.session_state["num_results"] = 5
    if "temperature" not in st.session_state:
        st.session_state["temperature"] = 0.7
    if "model_choice" not in st.session_state:
        st.session_state["model_choice"] = "auto"
    if "selected_index" not in st.session_state:
        st.session_state["selected_index"] = OPENSEARCH_INDEX
    if "text_field" not in st.session_state:
        st.session_state["text_field"] = "text"
    if "vector_field" not in st.session_state:
        st.session_state["vector_field"] = "embedding"
    if "use_custom_index" not in st.session_state:
        st.session_state["use_custom_index"] = False

    # Initialize OpenSearch client
    with st.spinner("Connecting to OpenSearch..."):
        client = get_opensearch_client()
    index_name = OPENSEARCH_INDEX

    # Ensure the index exists
    create_index(client)

    # Custom index selection checkbox
    st.session_state["use_custom_index"] = st.sidebar.checkbox(
        "🔧 Use Custom Index",
        value=st.session_state["use_custom_index"],
        help="Enable to select different OpenSearch index and field mappings"
    )
    
    # Show index and field selection only if checkbox is enabled
    if st.session_state["use_custom_index"]:
        available_indices = get_available_indices()
        if available_indices:
            current_index = st.session_state["selected_index"]
            if current_index not in available_indices:
                current_index = available_indices[0]
            
            st.session_state["selected_index"] = st.sidebar.selectbox(
                "📊 OpenSearch Index",
                options=available_indices,
                index=available_indices.index(current_index) if current_index in available_indices else 0,
                help="Select the OpenSearch index to search"
            )
            
            # Field selection based on selected index
            if st.session_state["selected_index"]:
                fields = get_index_fields(st.session_state["selected_index"])
                
                # Text field selection
                if fields["text_fields"]:
                    current_text_field = st.session_state["text_field"]
                    if current_text_field not in fields["text_fields"]:
                        current_text_field = fields["text_fields"][0]
                    
                    st.session_state["text_field"] = st.sidebar.selectbox(
                        "📝 Text Field",
                        options=fields["text_fields"],
                        index=fields["text_fields"].index(current_text_field) if current_text_field in fields["text_fields"] else 0,
                        help="Select the text field to search"
                    )
                else:
                    st.sidebar.warning("No text fields found")
                
                # Vector field selection
                if fields["vector_fields"]:
                    current_vector_field = st.session_state["vector_field"]
                    if current_vector_field not in fields["vector_fields"]:
                        current_vector_field = fields["vector_fields"][0]
                    
                    st.session_state["vector_field"] = st.sidebar.selectbox(
                        "🔢 Vector Field",
                        options=fields["vector_fields"],
                        index=fields["vector_fields"].index(current_vector_field) if current_vector_field in fields["vector_fields"] else 0,
                        help="Select the vector field for semantic search"
                    )
                else:
                    st.sidebar.info("No vector fields found - only keyword search available")
                    
                    # Add embeddings and fix mapping buttons when no vector fields exist
                    if st.button("✨ Enable Semantic Search", help="Add embeddings and configure knn_vector mapping"):
                        with st.spinner(f"Enabling semantic search for {st.session_state['selected_index']}..."):
                            try:
                                client = get_opensearch_client()
                                model = get_embedding_model()
                                
                                if not client or not model:
                                    st.error("Failed to initialize client or embedding model")
                                else:
                                    # Step 1: Fix vector mapping first
                                    import sys
                                    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
                                    from fix_vector_mapping import fix_vector_mapping
                                    
                                    success = fix_vector_mapping(st.session_state["selected_index"], "embedding", st.session_state["text_field"])
                                    if success:
                                        st.success("✅ Semantic search enabled successfully!")
                                        st.info("🔄 Refresh the page to see vector field options")
                                    else:
                                        st.error("Failed to enable semantic search")
                            except Exception as e:
                                st.error(f"Failed to enable semantic search: {str(e)}")
                                logger.error(f"Semantic search enablement failed: {e}")
        else:
            st.sidebar.error("No indices found")
    else:
        # Reset to defaults when custom index is disabled
        st.session_state["selected_index"] = OPENSEARCH_INDEX
        st.session_state["text_field"] = "text"
        st.session_state["vector_field"] = "embedding"
    
    # Sidebar settings for model selection, hybrid search toggle, result count, and temperature
    st.session_state["model_choice"] = st.sidebar.selectbox(
        "Response Mode",
        options=["auto", "openai", "ollama", "search_only"],
        index=["auto", "openai", "ollama", "search_only"].index(st.session_state["model_choice"]),
        help="Choose response mode: AI models or search-only to get document chunks"
    )
    
    # Add clear chat button
    if st.sidebar.button("🗑️ Clear Chat"):
        st.session_state["chat_history"] = []
        st.rerun()
    
    st.session_state["use_hybrid_search"] = st.sidebar.checkbox(
        "Enable RAG mode", value=st.session_state["use_hybrid_search"]
    )
    st.session_state["num_results"] = st.sidebar.number_input(
        "Number of Results in Context Window",
        min_value=1,
        max_value=10,
        value=st.session_state["num_results"],
        step=1,
    )
    st.session_state["temperature"] = st.sidebar.slider(
        "Response Temperature",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state["temperature"],
        step=0.1,
    )

    # Display logo or placeholder
    logo_path = "images/jamwithai_logo.png"
    if os.path.exists(logo_path):
        st.sidebar.image(logo_path, width=220)
        logger.info("Logo displayed.")
    else:
        st.sidebar.markdown("### Logo Placeholder")
        logger.warning("Logo not found, displaying placeholder.")

    # Sidebar headers and footer
    st.sidebar.markdown(
        "<h2 style='text-align: center;'>OpenSearch AI</h2>", unsafe_allow_html=True
    )
    st.sidebar.markdown(
        "<h4 style='text-align: center;'>Your Conversational Platform</h4>",
        unsafe_allow_html=True,
    )

    # Footer text
    st.sidebar.markdown(
        """
        <div class="footer-text">
            © 2025 OpenSearch AI
        </div>
        """,
        unsafe_allow_html=True,
    )
    logger.info("Sidebar configured with headers and footer.")

    # Display loading spinner at the top of the main content area
    with model_loading_placeholder.container():
        st.spinner("Loading models for chat...")

    # Load models if not already loaded
    if "embedding_models_loaded" not in st.session_state:
        with model_loading_placeholder:
            with st.spinner("Loading models..."):
                get_embedding_model()
                # Ensure Ollama model is available as fallback
                try:
                    ollama.chat(model=OLLAMA_MODEL_NAME, messages=[{"role": "user", "content": "test"}])
                    logger.info(f"Ollama model {OLLAMA_MODEL_NAME} is available")
                except Exception as e:
                    logger.warning(f"Ollama model not available: {e}")
                st.session_state["embedding_models_loaded"] = True
        logger.info("Models loaded.")
        model_loading_placeholder.empty()

    # Initialize chat history and processing flag
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "processing" not in st.session_state:
        st.session_state["processing"] = False

    # Display chat history
    for message in st.session_state["chat_history"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Process user input and generate response
    if prompt := st.chat_input("Type your message here..."):
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Add user message to history
        st.session_state["chat_history"].append({"role": "user", "content": prompt})
        
        # Generate and display assistant response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            response_text = ""
            final_response = ""
            thinking_shown = False
            response_stream = None  # Initialize response_stream
            
            # Handle search-only mode OR fallback to search when AI models unavailable
            if st.session_state["model_choice"] == "search_only":
                if st.session_state["use_hybrid_search"]:
                    embedding_model = get_embedding_model()
                    if embedding_model:
                        query_embedding = embedding_model.encode(prompt).tolist()
                        search_results = hybrid_search(
                            prompt, query_embedding, 
                            top_k=st.session_state["num_results"],
                            index_name=st.session_state.get("selected_index", OPENSEARCH_INDEX),
                            text_field=st.session_state.get("text_field", "text"),
                            vector_field=st.session_state.get("vector_field", "embedding")
                        )
                        
                        # Display search results as chunks
                        if search_results:
                            response_text = "**📄 Document Chunks Found:**\n\n"
                            for i, result in enumerate(search_results, 1):
                                text = result.get('_source', {}).get(st.session_state.get("text_field", "text"), '')
                                doc_name = result.get('_source', {}).get('document_name', 'Unknown')
                                score = result.get('_score', 0)
                                response_text += f"**Chunk {i}** (Score: {score:.2f}) - *{doc_name}*\n{text}\n\n---\n\n"
                        else:
                            response_text = "No relevant documents found."
                    else:
                        response_text = "Embedding model not available for search."
                else:
                    response_text = "Please enable RAG mode to search documents."
                
                response_placeholder.markdown(response_text)
                st.session_state["chat_history"].append({"role": "assistant", "content": response_text})
            else:
                # Try AI response first, fallback to search if models unavailable
                response_stream = generate_response_streaming(
                    prompt,
                    use_hybrid_search=st.session_state["use_hybrid_search"],
                    num_results=st.session_state["num_results"],
                    temperature=st.session_state["temperature"],
                    chat_history=st.session_state["chat_history"][:-1],
                    model_choice=st.session_state["model_choice"],
                    selected_index=st.session_state.get("selected_index", OPENSEARCH_INDEX),
                    text_field=st.session_state.get("text_field", "text"),
                    vector_field=st.session_state.get("vector_field", "embedding"),
                )
                
                # Check if AI response failed and fallback to search
                if response_stream is None:
                    if st.session_state["use_hybrid_search"]:
                        embedding_model = get_embedding_model()
                        if embedding_model:
                            query_embedding = embedding_model.encode(prompt).tolist()
                            search_results = hybrid_search(
                                prompt, query_embedding, 
                                top_k=st.session_state["num_results"],
                                index_name=st.session_state.get("selected_index", OPENSEARCH_INDEX),
                                text_field=st.session_state.get("text_field", "text"),
                                vector_field=st.session_state.get("vector_field", "embedding")
                            )
                            
                            if search_results:
                                response_text = "**⚠️ AI models unavailable. Showing document chunks:**\n\n"
                                for i, result in enumerate(search_results, 1):
                                    text = result.get('_source', {}).get(st.session_state.get("text_field", "text"), '')
                                    doc_name = result.get('_source', {}).get('document_name', 'Unknown')
                                    score = result.get('_score', 0)
                                    response_text += f"**Chunk {i}** (Score: {score:.2f}) - *{doc_name}*\n{text}\n\n---\n\n"
                                
                                response_placeholder.markdown(response_text)
                                st.session_state["chat_history"].append({"role": "assistant", "content": response_text})
                                response_stream = None  # Prevent further processing
                        
                        if response_stream is None and not search_results:
                            response_text = "No AI models or documents available."
                            response_placeholder.markdown(response_text)
                            st.session_state["chat_history"].append({"role": "assistant", "content": response_text})
            
            # Process AI response stream if available
            if response_stream is not None:
                for chunk in response_stream:
                    response_text += str(chunk)
                    
                    # Handle thinking tags
                    if "<think>" in response_text and "</think>" in response_text:
                        think_start = response_text.find("<think>") + 7
                        think_end = response_text.find("</think>")
                        thinking_text = response_text[think_start:think_end]
                        final_response = response_text[think_end + 8:].strip()
                        
                        if not thinking_shown:
                            with st.expander("🧠 AI Thinking Process", expanded=False):
                                st.markdown(thinking_text)
                            thinking_shown = True
                        
                        response_placeholder.markdown(final_response)
                    elif "<think>" in response_text:
                        response_placeholder.markdown('<div class="thinking-text">thinking...</div>', unsafe_allow_html=True)
                    else:
                        response_placeholder.markdown(response_text + "▌")
                
                # Final cleanup
                if final_response:
                    response_placeholder.markdown(final_response)
                    content_to_save = final_response
                else:
                    response_placeholder.markdown(response_text)
                    import re
                    content_to_save = re.sub(r'<think>.*?</think>', '', response_text, flags=re.DOTALL).strip()
                
                # Add assistant response to history
                if content_to_save:
                    st.session_state["chat_history"].append({"role": "assistant", "content": content_to_save})


# Main execution
if __name__ == "__main__":
    render_chatbot_page()

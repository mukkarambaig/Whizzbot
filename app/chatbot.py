# Requirements:
#  - Load embeddings
#  - Generate embeddings
#  - Initialize LLM chain
#  - Generate response
#  - Show the list of documents provided
#  - Save chat history
#  - Clear chat history

import streamlit as st

from src.utils.data_proprocessing import load_faiss_vectorstore, create_faiss_vectorstore

def load_knowledge_base():
    """Load the knowledge base."""
    st.session_state["docsearch"] = load_faiss_vectorstore()
    st.write("Knowledge base loaded successfully.")
    return st.session_state["docsearch"]


def reload_knowldge_base():
    """Reload the knowledge base."""
    create_faiss_vectorstore()
    st.session_state["docsearch"] = load_faiss_vectorstore()
    st.write("Knowledge base reloaded successfully.")
    return st.session_state["docsearch"]
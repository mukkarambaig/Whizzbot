# Third-party modules
import streamlit as st

# Custom modules
from chatbot import (load_knowledge_base, reload_knowledge_base,
                     clear_chat_history, save_chat_history)


def streamlit_app():
    """Main function to set up the Streamlit app."""
    st.set_page_config(page_title="💬 Whizzbridge HR Chatbot")

    with st.sidebar:
        st.title('💬 Whizzbridge HR Chatbot 13B')
        st.write("Version 2.0")

        if "docsearch" not in st.session_state.keys():
            load_knowledge_base()

        # Reload the knowledge base
        st.button("Reload Knowledge Base", on_click=reload_knowledge_base)

        # Clear the chat history
        st.button('Clear Chat History', on_click=clear_chat_history)

        # Save the chat history
        st.button('Save Chat History', on_click=save_chat_history)


if __name__ == "__main__":
    streamlit_app()

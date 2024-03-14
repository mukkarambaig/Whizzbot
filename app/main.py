# Build-in modules
import os
import time
from datetime import datetime

# Third-party modules
import streamlit as st
from dotenv import load_dotenv

# Custom modules
from service.chatbot import bot

# Debugging modules
from icecream import ic

# Initialize environment variables
load_dotenv()
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")

def show_documents():
    """Show the documents in the knowledge base."""
    with st.status("Documents in the Knowledge Base", expanded=True) as status:
        for i, file in enumerate(os.listdir(DOCUMENT_DIR)):
            status.write(f"{i+1}. {file}")


def save_chat_history():
    """
    Saves the chat history to a human-readable and timestamped file in the 'chats' directory.
    """
    # Ensure the 'chats' directory exists
    os.makedirs('chats', exist_ok=True)

    # Generate a human-readable and timestamped filename
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f'chats/chat_history_{timestamp}.txt'

    # Write the chat history to the file
    with open(filename, 'w') as file:
        for message in st.session_state.get('messages', []):
            # Use the message's timestamp if it exists, otherwise use the current timestamp
            msg_timestamp = message.get(
                "timestamp",
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            file.write(f"[{msg_timestamp}] {role}: {content}\n")

    # Notify the user
    st.success(f'Chat history saved as {filename}')


def clear_chat_history():
    """Clear the chat history."""
    st.session_state.messages = [{
        "role": "assistant",
        "content": "How may I assist you today?"
    }]



def streamlit_app():
    """Main function to set up the Streamlit app."""
    st.set_page_config(page_title="💬 Whizzbridge HR Chatbot")

    with st.sidebar:
        st.title('💬 Whizzbridge HR Chatbot 13B')
        st.write("Version 2.0")

        # Loading chatbot
        if "bot" not in st.session_state.keys():
            st.session_state['bot'] = bot()

        # Reload the knowledge base
        st.button("Reload Knowledge Base", on_click=st.session_state['bot'].reload_db)

        # Clear the chat history
        st.button('Clear Chat History', on_click=clear_chat_history)

        # Save the chat history
        st.button('Save Chat History', on_click=save_chat_history)

        # Show the documents in the knowledge base
        show_documents()

    # Store LLM generated responses
    if "messages" not in st.session_state.keys():
        st.session_state.messages = [{
            "role": "assistant",
            "content": "How may I assist you today?"
        }]

    # Display or clear chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input(placeholder="Type a message..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state['bot'].ask_model(prompt)
                placeholder = st.empty()
                full_response = ''
                for word in response:
                    for char in word:
                        full_response += char
                        placeholder.markdown(full_response)
                        time.sleep(0.005)
                message = {"role": "assistant", "content": full_response}
                st.session_state.messages.append(message)


if __name__ == "__main__":
    streamlit_app()

# Build-in modules
import os
import time
from datetime import datetime

# Third-party modules
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Custom modules
from utils.data_preprocessing import TextSplitter, VectorStoreManager
from utils.document_loader import read_unstructured_data
from utils.llm_model import BedrockManager

# Debugging modules
from icecream import ic

# Initialize environment variables
load_dotenv()
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR")
HF_API_KEY = os.getenv("HF_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))  # Providing default values
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "default-model-name")

# Initialize the objects
embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY, model_name=EMBEDDING_MODEL_NAME)
text_splitter = TextSplitter(CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME)
vectorstore_manager = VectorStoreManager(embedding_model)
bedrock_manager = BedrockManager()

# Define the chatbot prompt
prompt_template = ChatPromptTemplate.from_messages([
    ("system",
     """You are a document-based chatbot designed to assist HR professionals by providing precise answers drawn directly from a predefined set of documents. Your role is to analyze and extract relevant information from these documents to address inquiries related to HR practices, policies, and procedures.
                    When presented with a query:
                    1. Search and Respond: Thoroughly search the designated documents to find information directly related to the question. Your response should be strictly based on the document's content, providing specific, detailed answers.
                    2. Citation and Reference: For each response, cite the particular document and section where the information was found, guiding the user to the source and enhancing the response's credibility.
                    3. Handling Unknowns: If the query falls outside the scope of your documents, transparently acknowledge the limitation. Respond with "The information required is not available in my reference documents," and offer a courteous note of apology.
                    4. No External Data: Refrain from integrating any external information or personal insights into your responses. Your answers must be solely document-derived, maintaining the integrity and focus of the chatbot.
                    5. Confidentiality and Sensitivity: Respect the confidentiality of the documents. Do not disclose any sensitive or classified information in your responses.
                    6. Clarity and Brevity: Aim for responses that are clear, concise, and directly address the queries. Your communication should be easy to understand, avoiding unnecessary complexity or ambiguity.
                    7. Structured Responses: Present your answers in bullet points to enhance readability and make the information more accessible. This format will help HR professionals quickly grasp the essential points and apply them effectively.
                    Your mission is to assist HR professionals efficiently by providing accurate, document-based answers, respecting the chatbot's operational guidelines and the sensitive nature of HR-related inquiries."""
     ),
    ("human",
     "Provided Context: {context}\nUser want to know about: {user_input}"),
])

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


def reload_knowledge_base():
    """Reload the knowledge base."""
    # Create FAISS vectorstore
    text = read_unstructured_data(DOCUMENT_DIR)
    documents = text_splitter.split(text)
    vectorstore = vectorstore_manager.create_vectorstore(documents)

    # Save FAISS vectorstore
    vectorstore_manager.save_vectorstore(vectorstore, EMBEDDINGS_DIR)

    # Load FAISS vectorstore
    st.session_state['docsearch'] = vectorstore_manager.load_vectorstore(EMBEDDINGS_DIR)

    # Notify the user
    st.success(f'Knowledge Base reloaded from {DOCUMENT_DIR}')


def streamlit_app():
    """Main function to set up the Streamlit app."""
    st.set_page_config(page_title="💬 Whizzbridge HR Chatbot")

    with st.sidebar:
        st.title('💬 Whizzbridge HR Chatbot 13B')
        st.write("Version 2.0")

        if "docsearch" not in st.session_state.keys():
            st.session_state['docsearch'] = vectorstore_manager.load_vectorstore(EMBEDDINGS_DIR)
            
            st.success("Knowledge Base Loaded!", icon="✅") # Notify the user
        
        if "boto3_client" not in st.session_state.keys():
            st.session_state['boto3_client'] = bedrock_manager.initialize_boto3_client()
        
        if "retreiver" not in st.session_state.keys():
            st.session_state['retreiver'] = vectorstore_manager.get_vectorstore_retriever(st.session_state['docsearch'])
        
        if "rag_chain" not in st.session_state.keys():
            st.session_state['rag_chain'] = bedrock_manager.initialize_rag_chain(st.session_state['retreiver'], prompt_template, bedrock_manager.create_bedrock_instance(st.session_state['boto3_client']))
            ic(st.session_state['rag_chain'])
            st.success("Chat is ready to use!", icon="🚀")

        # Reload the knowledge base
        st.button("Reload Knowledge Base", on_click=reload_knowledge_base)

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

    if prompt := st.chat_input(placeholder="Type a message...", ):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Generate a new response if last message is not from assistant
    if st.session_state.messages[-1]["role"] != "assistant":
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = st.session_state['rag_chain'].invoke({"question": prompt})
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

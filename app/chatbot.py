# Build-in modules
import os
from datetime import datetime

# Third-party modules
import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_core.prompts import ChatPromptTemplate

# Custom modules
from src.utils.data_proprocessing import TextSplitter, VectorStoreManager
from src.utils.document_loader import read_unstructured_data
from src.utils.llm_model import BedrockManager

# Initialize environment variables
load_dotenv()
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR")
HF_API_KEY = os.getenv("HF_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))  # Providing default values
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "default-model-name")

embedding_model = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY, model_name=EMBEDDING_MODEL_NAME)
text_splitter = TextSplitter(CHUNK_SIZE, CHUNK_OVERLAP)
vectorstore_manager = VectorStoreManager(embedding_model)
bedrock_manager = BedrockManager()

chat_template = ChatPromptTemplate.from_messages(
    [
        ("system", """You are a document-based chatbot designed to assist HR professionals by providing precise answers drawn directly from a predefined set of documents. Your role is to analyze and extract relevant information from these documents to address inquiries related to HR practices, policies, and procedures.
                    When presented with a query:
                    1. Search and Respond: Thoroughly search the designated documents to find information directly related to the question. Your response should be strictly based on the document's content, providing specific, detailed answers.
                    2. Citation and Reference: For each response, cite the particular document and section where the information was found, guiding the user to the source and enhancing the response's credibility.
                    3. Handling Unknowns: If the query falls outside the scope of your documents, transparently acknowledge the limitation. Respond with "The information required is not available in my reference documents," and offer a courteous note of apology.
                    4. No External Data: Refrain from integrating any external information or personal insights into your responses. Your answers must be solely document-derived, maintaining the integrity and focus of the chatbot.
                    5. Confidentiality and Sensitivity: Respect the confidentiality of the documents. Do not disclose any sensitive or classified information in your responses.
                    6. Clarity and Brevity: Aim for responses that are clear, concise, and directly address the queries. Your communication should be easy to understand, avoiding unnecessary complexity or ambiguity.
                    7. Structured Responses: Present your answers in bullet points to enhance readability and make the information more accessible. This format will help HR professionals quickly grasp the essential points and apply them effectively.
                    Your mission is to assist HR professionals efficiently by providing accurate, document-based answers, respecting the chatbot's operational guidelines and the sensitive nature of HR-related inquiries."""),
        ("human", "Provided Context: {context}\nUser want to know about: {user_input}"),
    ]
)

bedrock_client = bedrock_manager.initialize_boto3_client()

def create_knowedge_base():
    """Create and save the FAISS vectorstore."""
    # Create FAISS vectorstore
    text = read_unstructured_data(DOCUMENT_DIR)
    documents = text_splitter.split(text)
    vectorstore = vectorstore_manager.create_vectorstore(documents)

    # Save FAISS vectorstore
    vectorstore_manager.save_vectorstore(vectorstore, EMBEDDINGS_DIR)

    # Notify the user
    st.success(f'Knowledge Base created and saved as {EMBEDDINGS_DIR}')    


def load_knowledge_base():
    """Load the FAISS vectorstore into the session state."""
    # Load FAISS vectorstore
    st.session_state['docsearch'] = vectorstore_manager.load_vectorstore(EMBEDDINGS_DIR)

    # Notify the user
    st.success("Knowledge Base Loaded!")


def prompt_engineering():
    pass


def initialize_llm_chain():
    """Initialize and return a RAG chain with the specified components."""
    retriever = vectorstore_manager.get_vectorstore_retriever(st.session_state['docsearch'])
    chain = bedrock_manager.initialize_rag_chain(retriever, prompt, bedrock_manager.bedrock_instance)


def generate_response():
    pass


def show_documents():
    pass


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
            msg_timestamp = message.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            role = message.get("role", "unknown").upper()
            content = message.get("content", "")
            file.write(f"[{msg_timestamp}] {role}: {content}\n")

    # Notify the user
    st.success(f'Chat history saved as {filename}')


def clear_chat_history():
    st.session_state.messages = [{
        "role": "assistant",
        "content": "How may I assist you today?"
    }]

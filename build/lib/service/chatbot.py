# Build-in modules
import os
from typing import List

# Third-party modules
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Custom modules
from utils.data_preprocessing import TextSplitter, VectorStoreManager
from utils.document_loader import read_unstructured_data
from utils.llm_model import BedrockManager
from utils.prompt_generator import prompt_generator


load_dotenv()
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR")
HF_API_KEY = os.getenv("HF_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))  # Providing default values
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "default-model-name")

EMBEDDING_MODEL = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY, model_name=EMBEDDING_MODEL_NAME)


def display(string: str) -> None:
            # Print the generated prompt for debugging or logging purposes
        print("===============DEBUGGING===============")
        print(string)
        print("===============DEBUGGING===============")

class bot:
    def __init__(self):
        self.text_splitter: TextSplitter = TextSplitter(CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME)
        self.text_chunks: List[str] = self.split_text()
        self.vectorstore_manager: VectorStoreManager = VectorStoreManager(EMBEDDING_MODEL)
        self.database = self.load_db()
        self.model_manager = BedrockManager()
        self.model_chain = self.model_manager.initialize_rag_chain()
        self.bot_conversation_memory = []
    
    def split_text(self):
        """Split the unstructured text data into chunks for processing."""
        text = read_unstructured_data(DOCUMENT_DIR)
        return self.text_splitter.split(text)
    
    def load_db(self):
        """Load the vectorstore from the disk."""
        return self.vectorstore_manager.load_vectorstore(EMBEDDINGS_DIR)
    
    def reload_db(self):
        """Reload the vectorstore from the disk."""
        self.database = self.vectorstore_manager.create_vectorstore(self.split_text())
        self.vectorstore_manager.save_vectorstore(self.database, EMBEDDINGS_DIR)
    
    def retrieve_context(self, question: str):
        """Retrieve relevant documents based on the user's question."""
        return self.vectorstore_manager.get_relevant_documents(self.database, question)
    
    def change_model_id(self, model_id):
        """Change the model_id of the Bedrock instance."""
        self.model_manager.change_model(model_id)
        self.model_chain = self.model_manager.initialize_rag_chain()
    
    def update_conversation_memory(self, user_input, model_response):
        """Update the conversation memory with the latest conversation turn, keeping only the last 5 turns."""
        if len(self.bot_conversation_memory) >= 5:
            # Remove the oldest conversation turn
            self.bot_conversation_memory.pop(0)
        # Add the new conversation turn
        self.bot_conversation_memory.append({'user': user_input, 'assistant': model_response})
    
    def extract_conversation_history(self):
        """Extract the conversation history from the bot's memory."""
        if not self.bot_conversation_memory:
            return "No conversation has taken place yet."
        
        history = []
        for conversation in self.bot_conversation_memory:
            user_line = f"user: {conversation['user']}"
            ai_line = f"AI: {conversation['assistant']}"
            history.append(f"{user_line}\n{ai_line}")
        return "\n".join(history)

    
    def extract_context(self, question):
        """Extract the context from the relevant documents based on the user's question."""
        docs = self.retrieve_context(question)
        SCORE = 0.9
        
        formatted_texts = [
            f" {doc.page_content if hasattr(doc, 'page_content') else 'No content available'}\nScore: {score}\n---\n"
            for __, (doc, score) in enumerate(docs, start=1) if score <= SCORE
        ]

        return "\n".join(formatted_texts).rstrip("---\n") if formatted_texts else "No relevant document found"

    # LECL Chain
    # TODO: Add memory to the model chain
    def ask_model(self, question: str):
        """Ask the model a question and return its response."""
        # Extract context based on the question
        context = self.extract_context(question)
        
        # Extract conversation history
        conversation_history = self.extract_conversation_history()
        
        # Generate a prompt using the extracted context, the question, and the conversation history
        prompt = prompt_generator(conversation_history, question, context)
        display(prompt)
        
        # For demonstration purposes, let's assume this is the response generation step
        response = self.model_chain.invoke(prompt)
        # response = "Testing response"  # Simulated response for demonstration
        
        # Update the conversation memory with the current interaction
        self.update_conversation_memory(question, response)
        
        # Now the updated conversation memory will be used in the next interaction
        return response

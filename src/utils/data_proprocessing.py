# Built-in modules
from typing import List
import os

# Third-party modules
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

# Custom modules
from document_loader import read_unstructured_data

# Initialize environment variables
load_dotenv()
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR")
HF_API_KEY = os.getenv("HF_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))  # Providing default values
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "default-model-name")

class TextSplitter:
    """Utility class to split text into chunks."""
    
    def __init__(self, chunk_size: int, chunk_overlap: int):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
    
    def split(self, text: str) -> List[str]:
        """Split the text into chunks."""
        return self.text_splitter.create_documents([text])

class VectorStoreManager:
    """Class to manage FAISS vectorstore operations."""
    
    def __init__(self, embedding_model):
        self.embedding_model = embedding_model
    
    def create_vectorstore(self, documents: List[str]):
        """Create and return a new FAISS vectorstore."""
        return FAISS.from_documents(documents=documents, embedding=self.embedding_model)
    
    def save_vectorstore(self, vectorstore, vectorstore_path: str):
        """Save the FAISS vectorstore to a path."""
        vectorstore.save_local(vectorstore_path)
    
    def load_vectorstore(self, vectorstore_path: str):
        """Load and return a FAISS vectorstore from a path."""
        return FAISS.load_local(vectorstore_path, embeddings=self.embedding_model)

def main():
    embedding_model = HuggingFaceInferenceAPIEmbeddings(
        api_key=HF_API_KEY, model_name=EMBEDDING_MODEL_NAME
    )
    text_splitter = TextSplitter(CHUNK_SIZE, CHUNK_OVERLAP)
    vectorstore_manager = VectorStoreManager(embedding_model)

    # Create FAISS vectorstore
    text = read_unstructured_data(DOCUMENT_DIR)
    documents = text_splitter.split(text)
    vectorstore = vectorstore_manager.create_vectorstore(documents)
    
    # Save FAISS vectorstore
    vectorstore_manager.save_vectorstore(vectorstore, EMBEDDINGS_DIR)

    # Load FAISS vectorstore
    loaded_vectorstore = vectorstore_manager.load_vectorstore(EMBEDDINGS_DIR)
    
    print("Done")

if __name__ == "__main__":
    main()

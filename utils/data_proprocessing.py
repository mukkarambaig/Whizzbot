from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma, FAISS
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings

from utils.document_loader import read_unstructured_data


embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key="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", model_name="sentence-transformers/all-mpnet-base-v1"
)

embeddings_path = "/home/mirza/repos/Whizzbot/data/faiss_embeddings"
data_path = "/home/mirza/repos/Whizzbot/data/documents"

def recursive_character_text_splitter(text: str, chunk_size: int = 1000, chunk_overlap: int = 0) -> List[str]:
    """
    Split the text into chunks of size `chunk_size` with overlap `chunk_overlap`.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    splits = text_splitter.create_documents(text)
    return splits

def create_faiss_vectorstore(embedding=embeddings_model):
    text = read_unstructured_data(data_path)

    documents = recursive_character_text_splitter(text)

    vectorstore = FAISS.from_documents(documents=documents, embedding=embedding)
    return vectorstore

def load_faiss_vectorstore(vectorstore_path: str = embeddings_path, embedding=embeddings_model):
    vectorstore = FAISS.load_local(vectorstore_path, embeddings=embedding)
    return vectorstore

def get_vectorstore_retriever(vectorstore):
    return vectorstore.as_retriever()

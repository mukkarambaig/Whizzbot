from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from document_reader import read_documents_from_directory
from langchain.text_splitter import CharacterTextSplitter
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
import os

vector_store_path = "data/faiss_embeddings"
data_path = "data/documents"
embeddings_model = HuggingFaceInferenceAPIEmbeddings(
    api_key="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", model_name="sentence-transformers/all-mpnet-base-v1"
)


def is_directory_empty(directory):
    # List the directory contents
    return not os.listdir(directory)


def load_vector_embeddings():
    if is_directory_empty(vector_store_path):
        return None

    docsearch = FAISS.load_local(vector_store_path, embeddings_model)
    docs = docsearch.similarity_search("What is the company's policy regarding employee leave entitlements?" , k=3)
    # for i, doc in enumerate(docs):
    #     print(f"Doc {i+1}: {doc.page_content} \n\n\n ")
    return docsearch


def create_text_chunks():
    text = read_documents_from_directory(data_path)
    text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1500,
                                          chunk_overlap=200, length_function=len)
    text_chucks = text_splitter.split_text(text)
    return text_chucks


def recursive_create_text_chunks():
    text = read_documents_from_directory(data_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500,
                                                   chunk_overlap=200, length_function=len)
    text_chucks = text_splitter.create_documents(text)
    return text_chucks


def create_vector_embeddings():
    text_chunks = create_text_chunks()
    print(len(text_chunks))
    print(text_chunks[:3])
    docsearch = FAISS.from_texts(text_chunks, embeddings_model)
    docsearch.save_local(vector_store_path)

def recursive_create_vector_embeddings():
    docs = recursive_create_text_chunks()
    print(len(docs))
    print(docs[:3])
    docsearch = FAISS.from_documents(docs, embeddings_model)
    docsearch.save_local(vector_store_path)

if __name__ == "__main__":
    # recursive_create_vector_embeddings()
    # # create_vector_embeddings()
    # load_vector_embeddings()
    print("Done")
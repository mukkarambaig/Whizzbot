import os
from tqdm import tqdm
from typing import List

from langchain_community.document_loaders import UnstructuredFileLoader

def read_unstructured_data(directory: str) -> str:
    """Reads unstructured data from all files in a specified directory and concatenates their contents.
    
    Args:
        directory (str): The path to the directory containing the documents.
        
    Returns:
        str: The concatenated content of all documents.
    """
    text_components = []
    for filename in tqdm(os.listdir(directory), desc="Reading files"):
        file_path = os.path.join(directory, filename)
        
        # Skip if it's a directory or unreadable file
        if not os.path.isfile(file_path):
            continue
        
        try:
            loader = UnstructuredFileLoader(file_path)
            docs = loader.load()
            if docs and hasattr(docs[0], 'page_content'):
                text_components.append(docs[0].page_content)
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")

    return '\n'.join(text_components)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    data_path = os.getenv("DOCUMENT_DIR")
    text = read_unstructured_data(data_path)
    print(text)

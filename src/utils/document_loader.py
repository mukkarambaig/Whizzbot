import os
from tqdm import tqdm
from typing import List
from icecream import ic

from langchain_community.document_loaders import UnstructuredFileLoader

def read_unstructured_data(directory: str) -> str:
    text = []
    for file in tqdm(os.listdir(directory)):
        file_path = ic(os.path.join(directory, file))
        loader = UnstructuredFileLoader(file_path)
        docs = loader.load()
        text.append(docs[0].page_content)

    return '\n'.join(text)

if __name__ == "__main__":
    data_path = "/home/mirza/repos/Whizzbot/data/documents"
    text = read_unstructured_data(data_path)
    print(text)
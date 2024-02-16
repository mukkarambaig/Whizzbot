'''
Helper functions to read data from documents
'''

# Importing required libraries
import os
from PyPDF2 import PdfReader
import pandas as pd
import docx


def read_pdf(file_path):
    ''' Read text from a PDF file '''
    with open(file_path, "rb") as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            text += pdf_reader.pages[page_num].extract_text()
    return text


def read_word(file_path):
    ''' Read text from a Word file '''
    doc = docx.Document(file_path)
    text = ""
    for paragraph in doc.paragraphs:
        text += paragraph.text + "\n"
    return text


def read_txt(file_path):
    ''' Read text from a text file '''
    with open(file_path, "r") as file:
        text = file.read()
    return text


def read_csv(file_path):
    ''' Read text from a CSV file '''
    df = pd.read_csv(file_path)
    return df.to_string()


def read_excel(file_path):
    ''' Read text from an Excel file '''
    df = pd.read_excel(file_path)
    return df.to_string()


def read_documents_from_directory(directory):
    ''' Read text from all documents in a directory '''
    file_handlers = {
        '.pdf': read_pdf,
        '.docx': read_word,
        '.txt': read_txt,
        '.csv': read_csv,
        '.xls': read_excel,
        '.xlsx': read_excel
    }

    texts = []
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        file_extension = os.path.splitext(filename)[1].lower()

        handler = file_handlers.get(file_extension)
        if handler:
            try:
                texts.append(handler(file_path))
            except Exception as e:
                print(f"Error reading {filename}: {e}")

    return '\n'.join(texts)
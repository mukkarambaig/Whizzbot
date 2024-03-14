# Build-in modules
import os
from typing import List

# Third-party modules
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain.prompts import ChatPromptTemplate

# Custom modules
from utils.data_preprocessing import TextSplitter, VectorStoreManager
from utils.document_loader import read_unstructured_data
from utils.llm_model import BedrockManager

load_dotenv()
DOCUMENT_DIR = os.getenv("DOCUMENT_DIR")
EMBEDDINGS_DIR = os.getenv("EMBEDDINGS_DIR")
HF_API_KEY = os.getenv("HF_API_KEY")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))  # Providing default values
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "20"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "default-model-name")

EMBEDDING_MODEL = HuggingFaceInferenceAPIEmbeddings(
    api_key=HF_API_KEY, model_name=EMBEDDING_MODEL_NAME)

# Define the chatbot prompt
prompt_template = ChatPromptTemplate.from_messages(
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
            ("human", "Provided Context: {context}\nUser want to know about: {question}"),
        ]
    )


class bot:
    def __init__(self):
        self.text_splitter: TextSplitter = TextSplitter(CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME)
        self.text_chunks: List[str] = self.split_text()
        self.vectorstore_manager: VectorStoreManager = VectorStoreManager(EMBEDDING_MODEL)
        self.database = self.load_db()
        self.model_manager = BedrockManager()
        self.model_chain = self.model_manager.initialize_qa_chain()
    
    def split_text(self):
        text = read_unstructured_data(DOCUMENT_DIR)
        return self.text_splitter.split(text)
    
    def load_db(self):
        return self.vectorstore_manager.load_vectorstore(EMBEDDINGS_DIR)
    
    def reload_db(self):
        self.database = self.vectorstore_manager.create_vectorstore(self.split_text())
        self.vectorstore_manager.save_vectorstore(self.database, EMBEDDINGS_DIR)
    
    def retrieve_context(self, question: str):
        return self.vectorstore_manager.get_relevant_documents(self.database, question)
    
    def prompt_engineering(self, context: str, question: str):
        return prompt_template.format_messages(context=context, question=question)
    
    def ask_model(self, question: str):
        context = self.retrieve_context(question)
        user_message = self.prompt_engineering(context, question)
        response = self.model_chain.invoke({"input_documents": context, "question": user_message})
        return response['output_text']
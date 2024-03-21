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

# TODO: Use a more enhance prompt to generate crisp and short answers!
# Define the chatbot prompt
# prompt_template = ChatPromptTemplate.from_messages(
#         [
#             ("system", """You are a document-based chatbot designed to assist HR professionals by providing precise, to-the-point answers drawn from a predefined set of documents. Your role is to analyze and extract only the most relevant information to address inquiries related to HR practices, policies, and procedures succinctly.
# When presented with a query:
# 1. Search and Respond: Efficiently search the designated documents to extract and provide information that is directly related to the question. Ensure your responses are concise, specific, and detailed, using bullet points for clarity.
#    - Example: If asked, "What is the company's policy on remote work?" your response should be:
#      - Remote work is allowed for roles deemed suitable by the department head.
#      - Employees must seek approval from their supervisor.
#      - The policy details are outlined in the Employee Handbook, Section 3.4.
# 2. Citation and Reference: Clearly cite the specific document and section where the information was found, enhancing the response's credibility and guiding the user to the source.
#    - Example: See Employee Handbook, Section 3.4 for full remote work policy details.
# 3. Handling Unknowns: If the query is beyond the scope of your documents, acknowledge this transparently, responding with, "The required information is not available in my reference documents." Include a polite note of apology.
#    - Example: I'm sorry, the information on freelance employment policies is not available in my reference documents.
# 4. No External Data: Your responses should be strictly based on the document's content without incorporating external information or personal insights, preserving the chatbot's integrity.
# 5. Confidentiality and Sensitivity: Maintain the confidentiality of the documents, avoiding the disclosure of any sensitive or classified information.
# 6. Clarity, Brevity, and Bullet Points: Prioritize clear, concise responses, and present them in bullet point format to directly address the queries and enhance readability. This approach helps HR professionals quickly find and apply the necessary information.
#    - Example: For a query about annual leave policies, respond with:
#      - Employees are entitled to 20 days of paid annual leave.
#      - Leave must be approved by the direct supervisor.
#      - Refer to the Employee Handbook, Section 5.2 for more details.
# 7. Structured Responses: Organize your answers in a structured manner, using bullet points to make the key points stand out, ensuring the information is accessible and actionable for HR professionals.
# Your mission is to support HR professionals efficiently by providing accurate, brief, and easily digestible, document-based answers, adhering to the operational guidelines and respecting the sensitive nature of HR inquiries."""),
#             ("human", "Provided Context: {context}\nUser want to know about: {question}"),
#         ]
#     )


class bot:
    def __init__(self):
        self.text_splitter: TextSplitter = TextSplitter(CHUNK_SIZE, CHUNK_OVERLAP, EMBEDDING_MODEL_NAME)
        self.text_chunks: List[str] = self.split_text()
        self.vectorstore_manager: VectorStoreManager = VectorStoreManager(EMBEDDING_MODEL)
        self.database = self.load_db()
        self.model_manager = BedrockManager()
        self.model_chain = self.model_manager.initialize_qa_chain()
        # self.model_chain = self.model_manager.initialize_memory_conversation_chain()
    
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
    
    def change_model_id(self, model_id):
        self.model_manager.change_model(model_id)
        self.model_chain = self.model_manager.initialize_qa_chain()
    
    def extract_context(self, question: str):
        docs = self.retrieve_context(question)
        formatted_text = ""
        for idx, doc in enumerate(docs, start=1):
            # Assuming each doc is an object with a page_content attribute
            content = doc.page_content if hasattr(doc, 'page_content') else "No content available"
            print(f"context {idx}: {content}\n---\n")
            formatted_text += f"context {idx}: {content}\n---\n"
        return formatted_text.strip("---\n")


    def ask_model(self, question: str):
        # context = self.retrieve_context(question)
        # print("===============Retrieved Context===============")
        # for i in context:
        #     print(i, end="\n--------------\n")
        # print("===============Retrieved Context===============")
        # print("\n\n\n")
        # user_message = self.prompt_engineering(context, question)
        # print("===============Prompt Engineered===============")
        # print(user_message)
        # print("===============Prompt Engineered===============")
        # response = self.model_chain.invoke({"input_documents": context, "question": user_message})
        # response = self.model_chain.invoke({"input_documents": context, "question": user_message})
        # return response['output_text']
        context = self.extract_context(question)
        print("===============Extracted Context===============")
        print(context)
        print("===============Extracted Context===============")
        prompt = prompt_generator(question, context)
        print("===============Prompt Engineered===============")
        print(prompt)
        print("===============Prompt Engineered===============")
        # TODO: Eliminate the docs variable and use the context variable
        # TODO: Add memory to the model chain
        docs = self.retrieve_context(question)
        response = self.model_chain.invoke({"input_documents": docs, "question": prompt})
        return response['output_text']
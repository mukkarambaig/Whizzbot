import os
from dotenv import load_dotenv
import boto3
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.bedrock import Bedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

class BedrockManager:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Initialize constants from environment variables
        self.service_name = os.getenv("SERVICE_NAME", "bedrock-runtime")
        self.region_name = os.getenv("AWS_REGION", "us-east-1")
        self.model_id = os.getenv("MODEL_ID", "meta.llama2-13b-chat-v1")
        self.streaming = os.getenv("STREAMING", "True").lower() == 'true'
        self.max_gen_len = int(os.getenv("MAX_GEN_LEN", "512"))
        self.temperature = float(os.getenv("TEMPERATURE", "0.2"))
        self.top_p = float(os.getenv("TOP_P", "0.9"))

        self.client = self.initialize_boto3_client()
        self.bedrock_instance = self.create_bedrock_instance(self.client)

    def initialize_boto3_client(self):
        """Initialize and return a boto3 client for the specified service and region."""
        return boto3.client(service_name=self.service_name, region_name=self.region_name)

    def create_bedrock_instance(self, client):
        """Create and return a Bedrock instance with the specified parameters."""
        return Bedrock(model_id=self.model_id, client=client, streaming=self.streaming,
                       callbacks=[StreamingStdOutCallbackHandler()],
                       model_kwargs={'max_gen_len': self.max_gen_len, 
                                     'temperature': self.temperature, 
                                     'top_p': self.top_p})

    def initialize_rag_chain(self, retriever, prompt, llm):
        """Initialize and return a RAG chain with the specified components."""
        return ({"context": retriever, "question": RunnablePassthrough()} 
                | prompt | llm | StrOutputParser())

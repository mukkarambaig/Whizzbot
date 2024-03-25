# Build-in modules
import os

# Third-party modules
from dotenv import load_dotenv
import boto3
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.bedrock import Bedrock
from langchain_core.output_parsers import StrOutputParser
from langchain.chains import LLMChain, ConstitutionalChain
from langchain.chains.constitutional_ai.models import ConstitutionalPrinciple

# Custom modules
from utils.prompt_generator import prompt_template_generator

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
        self.store = {} 
        self.client = self.initialize_boto3_client()
        self.bedrock_instance = self.create_bedrock_instance(self.client)

    def initialize_boto3_client(self):
        """Initialize and return a boto3 client for the specified service and region."""
        return boto3.client(service_name=self.service_name, region_name=self.region_name)
    
    def change_model(self, model_id):
        """Change the model_id of the Bedrock instance."""
        self.model_id = model_id
        self.bedrock_instance = self.create_bedrock_instance(self.client)

    def create_bedrock_instance(self, client):
        """Create and return a Bedrock instance with the specified parameters."""
        return Bedrock(model_id=self.model_id, client=client, streaming=self.streaming,
                       callbacks=[StreamingStdOutCallbackHandler()],
                       model_kwargs={'max_gen_len': self.max_gen_len, 
                                     'temperature': self.temperature, 
                                     'top_p': self.top_p})

    def initialize_rag_chain(self):
        """Initialize and return a RAG chain with the specified components."""
        return (self.bedrock_instance | StrOutputParser())
    
    def initialize_llm_chain(self):
        """Initialize and return a LLM chain with the specified components."""
        return LLMChain(llm=self.bedrock_instance, prompt=prompt_template_generator())
    
    # TODO: The llm argument require LLMChain instance
    def constitutional_chain(self):
        """Return a constitutional chain with the specified components."""
        principles = ConstitutionalChain.get_principles(["uo-ethics-1"])
        constitutional_chain = ConstitutionalChain.from_llm(
            llm = self.bedrock_instance,
            chain = self.initialize_llm_chain(),
            constitutional_principles = principles,
            verbose = True,
        )
        return constitutional_chain
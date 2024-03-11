import os
from dotenv import load_dotenv
import boto3
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.bedrock import Bedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

# Load environment variables
load_dotenv()

# Constants
SERVICE_NAME = os.getenv("SERVICE_NAME")
REGION_NAME = os.getenv("AWS_REGION")
MODEL_ID = os.getenv("MODEL_ID")
STREAMING = os.getenv("STREAMING", "True").lower() == 'true'  # Convert to boolean
MAX_GEN_LEN = int(os.getenv("MAX_GEN_LEN",))  # Convert to integer
TEMPERATURE = float(os.getenv("TEMPERATURE"))  # Convert to float
TOP_P = float(os.getenv("TOP_P"))  # Convert to float

def initialize_boto3_client(service_name: str = SERVICE_NAME, region_name: str = REGION_NAME):
    """Initialize and return a boto3 client for the specified service and region."""
    return boto3.client(service_name=service_name, region_name=region_name)

def create_bedrock_instance(client, model_id: str = MODEL_ID, streaming: bool = STREAMING,
                            max_gen_len: int = MAX_GEN_LEN, temperature: float = TEMPERATURE,
                            top_p: float = TOP_P):
    """Create and return a Bedrock instance with the specified parameters."""
    return Bedrock(model_id=model_id, client=client, streaming=streaming,
                   callbacks=[StreamingStdOutCallbackHandler()],
                   model_kwargs={'max_gen_len': max_gen_len, 'temperature': temperature, 'top_p': top_p})

def format_docs(docs):
    """Format and return document contents."""
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag_chain(retriever, prompt, llm):
    """Initialize and return a RAG chain with the specified components."""
    return ({"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser())
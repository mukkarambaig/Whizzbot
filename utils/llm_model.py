import boto3
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.llms.bedrock import Bedrock
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def initialize_boto3_client(service_name: str = "bedrock-runtime", region_name: str = "us-east-1"):
    return boto3.client(service_name=service_name, region_name=region_name)

def create_bedrock_instance(client, model_id: str = "meta.llama2-13b-chat-v1", streaming: bool = True, max_gen_len: int = 512, temperature: float = 0.2, top_p: float = 0.9):
    """
    Creates and returns a Bedrock instance with specified parameters.

    :param client: The bedrock client instance.
    :param model_id: The model ID to use.
    :param streaming: Boolean indicating if streaming is enabled.
    :param max_gen_len: Maximum generation length.
    :param temperature: Temperature setting for generation.
    :param top_p: Top-p setting for generation.
    :return: A Bedrock instance.
    """
    return Bedrock(model_id=model_id, client=client, streaming=streaming, 
                   callbacks=[StreamingStdOutCallbackHandler()], 
                   model_kwargs={'max_gen_len': max_gen_len, "temperature": temperature, "top_p": top_p})

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def initialize_rag_chain(retriever, prompt, llm):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def testing_rag_chain(prompt, llm):
    return (
        {"question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
'''
WhizzBot singleton class
'''

# Import the required libraries
import time
import logging

from document_reader import read_documents_from_directory
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain_community.embeddings import HuggingFaceEmbeddings
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer

# Create a logger
logger = logging.getLogger('Logs')
logger.setLevel(logging.DEBUG)  # Set the minimum log level

# Create a file handler for outputting logs to a file
file_handler = logging.FileHandler('application.log')
file_handler.setLevel(logging.DEBUG)  # Can be set to a different level if needed

# Create a console handler for outputting logs to the console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Define the formatter and set it for both handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)


def timeit(func):
    ''' Decorator to measure the time a function takes '''
    def timed(*args, **kw):
        start_time = time.time()
        # Call the original function
        result = func(*args, **kw)
        end_time = time.time()
        # Log the time it took
        logger.info(f"*** {func.__name__} took {end_time - start_time:.3f} seconds")
        return result

    return timed


@timeit
def get_chain(model_path):
    ''' Function to get the language model chain '''
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)
        pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)
        llm = HuggingFacePipeline(pipeline=pipe)
        return load_qa_chain(llm, chain_type="stuff")

    except Exception as e:
        logger.error(f"Error in get_chain: {e}")
        return None


class WhizzBot:
    ''' Singleton class for the WhizzBot '''
    __instance = None


    def __new__(cls):
        ''' Singleton instance '''
        if cls.__instance is None:
            cls.__instance = super(WhizzBot, cls).__new__(cls)
        return cls.__instance

    def __init__(self):
        ''' Initialize the WhizzBot '''
        self.model = get_chain("FineTuned-TinyLlama-1.1B-Chat-v1.0")
        if self.model is None:
            logger.error("Model initialization failed")
            raise RuntimeError("Model initialization failed")

        self.embedding_model = HuggingFaceEmbeddings()
        self.data_path = "data"
        self.embeddings_file = "faiss_embeddings"
        self.docsearch = None
        try:
            self.load_embeddings()
        except Exception as e:
            logger.warning(f"Embeddings loading failed: {e}")

    @timeit
    def load_embeddings(self):
        ''' Load embeddings from file '''
        try:
            logger.info("*** Loading existing embeddings")
            self.docsearch = FAISS.load_local(self.embeddings_file, self.embedding_model)

        except Exception as e:
            logger.error(f"Error in load_embeddings: {e}")
            raise

    @timeit
    def _create_embeddings(self):
        ''' Create embeddings from the data '''
        try:
            text_chunks = self._text_chunks()
            self.docsearch = FAISS.from_texts(text_chunks, self.embedding_model)
            self.docsearch.save_local(self.embeddings_file)
            logger.info("*** Embeddings saved")

        except Exception as e:
            logger.error(f"Error in _create_embeddings: {e}")
            raise

    def _read_text(self):
        ''' Read text from the data path '''
        try:
            text = read_documents_from_directory(self.data_path)
            return text
        except Exception as e:
            logger.error(f"Error in _read_text: {e}")
            raise

    def _text_chunks(self):
        ''' Split the text into chunks '''
        try:
            text = self._read_text()
            char_text_splitter = CharacterTextSplitter(separator=" ", chunk_size=1000,
                                                       chunk_overlap=200, length_function=len)
            text_chunks = char_text_splitter.split_text(text)
            return text_chunks
        except Exception as e:
            logger.error(f"Error in _text_chunks: {e}")
            raise

    def set_data_path(self, new_path):
        ''' Set the data path and reinitialize the embeddings '''
        if new_path != self.data_path:
            logger.info(f"*** Updating data path to {new_path}")
            self.data_path = new_path
            try:
                self._create_embeddings()
                logger.info("*** Data path updated and embeddings reinitialized")
            except Exception as e:
                logger.error(f"Error in set_data_path: {e}")
                raise

    def get_data_path(self):
        ''' Get the data path '''
        return self.data_path

    @timeit
    def find_docs(self, query):
        ''' Find the most similar documents to the query '''
        docs = self.docsearch.similarity_search(query, k=1)
        return docs

    @timeit
    def predict(self, query, temp, top_k, top_p, max_length):
        ''' Make a prediction using the model '''
        try:
            docs = self.find_docs(query)  # Assuming find_docs is a method of the bot class
            logger.info(f"*** Starting prediction")
            response = self.model.invoke({"input_documents": docs, "question": query, "max_length": max_length,
                                          "temperature": temp, "top_k": top_k, "top_p": top_p})
            return response['output_text']
        except Exception as e:
            logger.error(f"Error in predict: {e}")
            raise

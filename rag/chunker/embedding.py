from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
import numpy as np

def encode(model="BAAI/bge-m3"):
    documents = SimpleDirectoryReader("../data").load_data()
    # bge-base embedding model
    Settings.embed_model = HuggingFaceEmbedding(model_name=model)
    # ollama
    Settings.llm = Ollama(model="llama3", request_timeout=360.0)
    index = VectorStoreIndex.from_documents(
        documents,
    )
    return np.array(index)

# https://docs.llamaindex.ai/en/stable/understanding/storing/storing/ - про хранение индексов
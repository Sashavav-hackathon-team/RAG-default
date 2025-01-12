import chromadb
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import StorageContext
from llama_index.core import Settings

model = "BAAI/bge-m3"
Settings.embed_model = HuggingFaceEmbedding(model_name=model)
# ollama
Settings.llm = Ollama(model="llama3", request_timeout=360.0)

# load some documents
documents = SimpleDirectoryReader("prepared").load_data()

# initialize client, setting path to save data
db = chromadb.PersistentClient(path="chroma_db")

# create collection
chroma_collection = db.get_or_create_collection("quickstart")

# assign chroma as the vector_store to the context
vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

# create your index
index = VectorStoreIndex.from_documents(
    documents, storage_context=storage_context
)

# create a query engine and query
#query_engine = index.as_query_engine()
#response = query_engine.query("Кто такая майа санду")
#print(response)

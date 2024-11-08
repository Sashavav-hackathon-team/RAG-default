import chromadb
from llama_index.core import VectorStoreIndex, get_response_synthesizer, StorageContext, Settings
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.vector_stores.chroma import ChromaVectorStore
from openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import llama_index.llms.openai


def deprecated_find_best_chunks(q: str, k: int) -> str:
    db = chromadb.PersistentClient(path="prepared/chroma_db")

    # get collection
    chroma_collection = db.get_or_create_collection("quickstart")

    # assign chroma as the vector_store to the context
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-m3")
    Settings.embed_model = embed_model
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        storage_context=storage_context,
    )

    llm = llama_index.llms.openai.OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    #llm = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    # Create the query engine
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=k,
    )

    '''response_synthesizer = get_response_synthesizer(response_mode=ResponseMode.COMPACT)
    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=response_synthesizer
    )

    # Query the index
    #response = query_engine.query("Write an email to the user given their background information.")
    #print(response)

    # Point to the local server
    '''
    query_engine = index.as_query_engine(llm=llm)
    response = query_engine.query(q)
    return response.response




from openai import OpenAI

# Set the base URL to your LM Studio server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_embeddings(texts, model="BAAI/bge-m3"):
    # Ensure texts are properly formatted
    texts = [text.replace("\n", " ") for text in texts]
    return client.embeddings.create(input=texts, model=model).data

# Example usage
texts = ["What is BGE M3?", "Definition of BM25"]
embeddings = get_embeddings(texts)
print(embeddings)
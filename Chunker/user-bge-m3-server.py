from openai import OpenAI

# Set the base URL to your LM Studio server
client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

def get_embeddings(texts, model="BAAI/bge-m3"):
    # Ensure texts are properly formatted
    texts = [text.replace("\n", " ") for text in texts]
    return client.embeddings.create(input=texts, model=model).data


import numpy as np
import scipy.spatial.distance as ds
import torch.nn as nn
import torch

# Example usage
texts = ["What is BGE M3?", "Definition of BM25"]
embeddings = get_embeddings(texts)
vector_1 = np.array(embeddings[0].embedding)
input1 = torch.tensor(embeddings[0].embedding)
texts = ["How does BGE M3 work?", "Main purpose of BM25"]
embeddings = get_embeddings(texts)
#print(*embeddings[0].embedding)

vector_2 = np.array(embeddings[0].embedding)

dis = ds.cosine(vector_1, vector_2)
print(dis)
cos = nn.CosineSimilarity(dim=0, eps=1e-6)

input2 = torch.tensor(embeddings[0].embedding)
output = cos(input1, input2)
print(output)
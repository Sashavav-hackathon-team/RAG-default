import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI
import scipy.spatial.distance as ds
import torch.nn as nn
import torch


def RecursiveChunker(name: str, chunk_size=200) -> str:
    f = open("../data/" + name, "r", encoding="utf-8")
    s = f.read()
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=200,
        chunk_overlap=20,
        length_function=len,
        is_separator_regex=False,
    )
    new_path = "../data/chunked_" + name
    docs = text_splitter.split_text(s)
    pf = open(new_path, "w", encoding="utf-8")
    for doc in docs:
        pf.write(doc + "\n" + "-" * 30 + "\n")
    return new_path


def LLMChanker(name: str) -> str:
    # Point to the local server
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")

    RULES = (
        "Always answer on russian. For each text, that i give you you should respond me with the text, that has only"
        " russian letters, comma and dot sign. You should semantic split this text on parts and return them, "
        "you should not respond with any information that is not included in that text, you should avoid all"
        " unnecessary words and constructions, you should focus only on your task, you will be tipped with 100$"
        " if you make everything like i said")

    def ask_question(question):
        completion = client.chat.completions.create(
            model="model-identifier",
            messages=[
                {"role": "system", "content": RULES},
                {"role": "user", "content": question}
            ],
            temperature=0.7,
        )

        return completion.choices[0].message.content

    f = open("../data/" + name, "r", encoding='utf-8')
    s = f.read()
    s = s.replace("\n", " ")
    new_path = "../data/chunked_" + name
    pf = open(new_path, "w", encoding='utf-8')
    for i in range(100, len(s) - 100, 300):
        answer = ask_question(s[i - 100:i + 400])
        pf.write(answer)
    return new_path


def get_embeddings(texts, model="BAAI/bge-m3"):
    client = OpenAI(base_url="http://localhost:1234/v1", api_key="lm-studio")
    # Ensure texts are properly formatted
    texts = [text.replace("\n", " ") for text in texts]
    return client.embeddings.create(input=texts, model=model).data


def search_in_chunks(chunk_vectors: np.array, query: np.array, k=5) -> np.array:
    res = np.empty((k, len(query)))
    best_indexes = [-1] * k
    for x in chunk_vectors:
        if len(x) != len(query):
            raise ValueError("Vector lengths do not match")

        # Example usage
        texts = x
        embeddings = get_embeddings(texts)
        input1 = torch.tensor(embeddings[0].embedding)
        texts = query
        embeddings = get_embeddings(texts)

        cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        input2 = torch.tensor(embeddings[0].embedding)
        output = cos(input1, input2)
        if output > min(best_indexes):
            i = best_indexes.index(min(best_indexes))
            best_indexes[i] = output
            res[i] = x
    return res

import json
import os
from pydantic import BaseModel
from typing import List, Optional

import requests
import numpy as np
from predictionguard import PredictionGuard

client = PredictionGuard()

def cosine_similarity(a, b):
    """Calculates the cosine similarity between two vectors."""
    dot_product = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    return dot_product / (norm_a * norm_b)

class RetrievalRequest(BaseModel):
    table: str
    query: str


class LLMConfig(BaseModel):
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.1


class AnswerRequest(BaseModel):
    retrieval: RetrievalRequest
    llm: Optional[LLMConfig] = LLMConfig(
        max_tokens=500,
        temperature=0.1
    )

def rag_answer(answer_request):

    url = os.environ.get("RAG_API_URL") + '/answers'

    payload = json.dumps(answer_request.dict())
    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()

def embed(query):
    """
    Function to embed the given query using embModel.

    Args:
    query (str): The input query to be embedded.

    Returns:
    numpy array: The embedded representation of the input query.
    """
    response = client.embeddings.create(
        model="multilingual-e5-large-instruct",
        input=query,
        truncate=True
    )
    return response["data"][0]["embedding"]

def run_table_test(table):

    # Loop over keys in the table dictionary and
    # embed the value.
    embs = {}
    for key, value in table.items():
        embs[key] = embed(value)

    # Loop over the keys in the table dictionary, get
    # the rag answers and emb them.
    embs_rag = {}
    for key, value in table.items():
        answer_request = AnswerRequest(retrieval=RetrievalRequest(
            table="testtxt", 
            query=key,
            hyde=True))
        response = rag_answer(answer_request)
        embs_rag[key] = embed(response['answer'])

    # Loop over the keys in the table dictionary, 
    # compare the embs with embs_rag using cosine similarity
    # and save the results in an array
    results = []
    for key, value in table.items():
        similarity = cosine_similarity(embs[key], embs_rag[key])
        results.append(similarity)

    # Print out the max, min, and mean of the results
    print("Max:", max(results))
    print("Min:", min(results))
    print("Mean:", np.mean(results))


if __name__ == "__main__":

    # Load the test fixture.
    with open('fixture.json', 'r') as f:
        fixture = json.load(f)

    run_table_test(fixture)
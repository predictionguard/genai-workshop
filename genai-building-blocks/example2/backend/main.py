from pydantic import BaseModel
from typing import List, Optional
import uuid

from fastapi import FastAPI, HTTPException
import pandas as pd
import lancedb
from langchain.prompts import PromptTemplate
import pyarrow as pa
from predictionguard import PredictionGuard

client = PredictionGuard()
app = FastAPI()


#--------------------------#
#         Config           #
#--------------------------#

default_chunk_size = 300
maximum_chunk_size = 1000
maximum_num_docs = 20
maximum_chunk_count = 5

INVALID_DOCS_ERROR = "Invalid docs uploads. Maximum number of documents is %d and maximum chunk size is %d characters." % (maximum_num_docs, maximum_chunk_size)
INVALID_SEARCH_ERROR = "Invalid search. Maximum limit of return docs is %d and maximum query size is %d characters." % (maximum_num_docs, maximum_chunk_size)
INVALID_REMOVE_TABLE_ERROR = "Table not found."
INVALID_SEARCH_TABLE_ERROR = "Table not found."
INVALID_ANSWER_TABLE_ERROR = "Table not found."
INVALID_CHUNK_COUNT_ERROR = "Invalid chunk count. Maximum number of chunks is %d." % maximum_chunk_count


#------------------------------#
# Vector DB, Retrieval setup   #
#------------------------------#

# local path of the vector db
uri = ".lancedb"
db = lancedb.connect(uri)
schema = pa.schema([
   pa.field("vector", pa.list_(pa.float32(), list_size=1024)),
   pa.field("text", pa.string()),
   pa.field("id", pa.string()),
   pa.field("metadata", pa.string())
])


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


def batch_embed(batch):
    """
    A function to embed a batch of strings using the AI21 LLM embeddings model.

    Args:
        batch (list): A list of strings to be embedded.

    Returns:
        list: A list of embeddings for the input strings.
    """
    embeddings = []
    if len(batch) < 10:
        response = client.embeddings.create(
            model="multilingual-e5-large-instruct",
            input=batch,
            truncate=True
        )
        for idx in range(0, len(response["data"])):
            for emb in response["data"]:
                if emb['index'] == idx:
                    embeddings.append(emb['embedding'])
    else:
        smaller_batches = [batch[i:i+10] for i in range(0, len(batch), 10)]
        embeddings = []
        for smaller_batch in smaller_batches:
            response = client.embeddings.create(
                model="multilingual-e5-large-instruct",
                input=smaller_batch,
                truncate=True
            )
            for idx in range(0, len(response["data"])):
                for emb in response["data"]:
                    if emb['index'] == idx:
                        embeddings.append(emb['embedding'])

    return embeddings

    
def validate_docs(docs):
    """
    Validate the provided docs and user_chunk_size.

    Args:
        docs (list): The list of documents to validate.
        user_chunk_size (int): The size of user's chunk.

    Returns:
        bool: True if the validation passes, False otherwise.
    """
    if len(docs) > 20:
       return False
    for d in docs:
        if len(d.doc) > maximum_chunk_size:
            return False
    return True

    
def add_docs(upload_docs_request):
    """
    A function to add documents to a database and return the created document UUIDs.
    Takes an upload_docs_request object as input.
    Returns a list of document UUIDs.
    """

    # Get the embeddings for the documents loaded.
    embeddings = batch_embed([doc.doc for doc in upload_docs_request.docs])
   
    # Loop over the docs and create uuids for each one.
    docs_load = []
    for i, doc in enumerate(upload_docs_request.docs):
        docs_load.append({
           "id": str(uuid.uuid4()),
           "text": doc.doc,
           "metadata": doc.metadata,
           "vector": embeddings[i]
        })

    # Insert the docs.
    table = db.open_table(upload_docs_request.table)
    table.merge_insert("text").when_matched_update_all().when_not_matched_insert_all().execute(docs_load)
    table.create_fts_index("text", replace=True)

    return docs_load


class RetrievalRequest(BaseModel):
    table: str
    query: str
    chunk_count: Optional[int] = 1


def search(retrieval_request):
    """
    Performs a search on the specified table using the provided query and returns the results. 

    Parameters:
        query (str): The search query.
        table_name (str): The name of the table to search.
        limit (int): The maximum number of results to return.

    Returns:
        list: A list of dictionaries containing the search results.
    """

    query_doc = retrieval_request.query

    # Open the table.
    table = db.open_table(retrieval_request.table)

    # Search the table.
    results = table.search(embed(query_doc)).limit(retrieval_request.chunk_count).to_pandas()
    print(results.columns)
    print(results.head())
    
    # Clearn up the results.
    results.drop(columns=['vector'], inplace=True)
    results.sort_values(by=['_distance'], inplace=True, ascending=True)
    return results


#--------------------------#
# Prompt, LLM setup        #
#--------------------------#

qa_system = "Read the context below and respond with an answer to the question. If the question cannot be answered based on the context alone or the context does not explicitly say the answer to the question, write \"Sorry I had trouble answering this question, based on the information I found.\""

qa_template = """Context: "{context}"
 
Question: "{query}"

Answer: """
 
qa_prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=qa_template
)


def rag_answer(answer_request):
  """
  Function to generate a response to an answer request using a pretrained model.
  
  Args:
      answer_request (AnswerRequest): An object containing the table name, query, max_tokens, and temperature.

  Returns:
      dict: A dictionary containing the generated answer, injected document, injected document ID, and distance.
  """

  # Execute the retrieval.
  results = search(answer_request.retrieval)

  # Sort and process the results.
  results.sort_values(by=['_distance'], inplace=True, ascending=True)
  if len(results) >= answer_request.retrieval.chunk_count:
      doc_use = "\n\n".join(results['text'].values[:answer_request.retrieval.chunk_count])
      idx = results['id'].values[:answer_request.retrieval.chunk_count]
      distance = results['_distance'].values[:answer_request.retrieval.chunk_count]
      metadata = results['metadata'].values[:answer_request.retrieval.chunk_count]
  else:
    doc_use = "\n\n".join(results['text'].values)
    idx = results['id'].values
    distance = results['_distance'].values
    metadata = results['metadata'].values
 
  # Augment the prompt with the context
  prompt = qa_prompt.format(context=doc_use, query=answer_request.retrieval.query)
 
  # Get a response
  result = client.chat.completions.create(
      model="Hermes-3-Llama-3.1-8B",
      messages=[
          {"role": "system", "content": qa_system},
          {"role": "user", "content": prompt}
      ],
      max_tokens=answer_request.llm.max_tokens,
      temperature=answer_request.llm.temperature
  )
 
  return {
      "answer": result["choices"][0]["message"]["content"],
      "injected_context": doc_use,
      "injected_doc_ids": idx.tolist(),
      "distances": distance.tolist(),
      "metadata": metadata.tolist()
  }


#---------------------------#
#        Routes             #
#---------------------------#

@app.get("/")
async def root():
    """
    A function that handles requests to the root URL. It is an asynchronous 
    function with no parameters and it returns a dictionary with a "message" 
    key containing the value "Hello World".
    """
    return {"message": "Hello World"}


class Doc(BaseModel):
    doc: str
    metadata: Optional[str] = None


class UploadDocsRequest(BaseModel):
    table: str
    docs: List[Doc]


@app.post("/docs")
async def upload_docs(upload_docs_request: UploadDocsRequest):
    """
    Uploads the provided docs to the LanceDB table, after validating the docs and creating the table if it doesn't exist. 
    Parameters:
        upload_docs_request: UploadDocsRequest - the request object containing the docs and chunk size.
    Returns:
        return_info: the information about the uploaded docs.
    """

    # Validate the docs for upload.
    valid = validate_docs(
       upload_docs_request.docs
    )
    if not valid:
       raise HTTPException(status_code=400, detail=INVALID_DOCS_ERROR)

    # Create the LanceDB table if it doesn't exist yet.
    tables = db.table_names()
    if upload_docs_request.table not in tables:
       db.create_table(upload_docs_request.table, schema=schema)

    # Embed and add the docs to the table.
    return_info = add_docs(upload_docs_request)

    return return_info


class RemoveDocsRequest(BaseModel):
    doc_ids: List[str]
    table: str


@app.delete("/docs")
async def delete_docs(remove_request: RemoveDocsRequest):
    """
    Delete docs from the specified table based on the remove request.
    """
    
    # Validate the table.
    tables = db.table_names()
    if remove_request.table not in tables:
       raise HTTPException(status_code=404, detail=INVALID_REMOVE_TABLE_ERROR)

    # Delete the docs.
    filter_ids = ", ".join(["'%s'" % id for id in remove_request.doc_ids])
    table = db.open_table(remove_request.table)
    table.delete(f"id IN ({filter_ids})")

    return


class AddTableRequest(BaseModel):
    table: str


@app.post("/tables")
async def add_table(add_table_request: AddTableRequest):
    """
    Asynchronous function to add a table to the database.

    Parameters:
        add_table_request (AddTableRequest): The request object containing the details of the table to be added.

    Returns:
        None
    """
    tables = db.table_names()
    if add_table_request.table not in tables:
       db.create_table(add_table_request.table, schema=schema)
    return


class RemoveTableRequest(BaseModel):
    table: str


@app.delete("/tables")
async def add_table(remove_table_request: RemoveTableRequest):
    """
    Delete a table from the database.

    Args:
        remove_table_request (RemoveTableRequest): The request object containing the table to be removed.

    Returns:
        None
    """
    tables = db.table_names()
    if remove_table_request.table in tables:
       db.drop_table(remove_table_request.table)
    return


@app.post("/retrievals")
async def search_db(retrieval_request: RetrievalRequest):
    """
    A function to handle the search request from the client, validate the request, 
    perform the search in the database, and return the search results.
    Parameters:
        search_request: SearchRequest - the search request object containing query, table, and num_results.
    Returns:
        return_info: SearchResult - the search result information.
    """

    # Make sure the table exists.
    tables = db.table_names()
    if retrieval_request.table not in tables:
       raise HTTPException(status_code=404, detail=INVALID_SEARCH_TABLE_ERROR)

    # Validate the request.
    if len(retrieval_request.query) > maximum_chunk_size or retrieval_request.chunk_count > maximum_chunk_count:
       raise HTTPException(status_code=400, detail=INVALID_SEARCH_ERROR)
    
    # Perform the search.
    results = search(retrieval_request)

    return results.to_dict('records')


class LLMConfig(BaseModel):
    max_tokens: Optional[int] = 500
    temperature: Optional[float] = 0.1


class AnswerRequest(BaseModel):
    retrieval: RetrievalRequest
    llm: Optional[LLMConfig] = LLMConfig(
        max_tokens=500,
        temperature=0.1
    )


@app.post("/answers")
async def answer_generation(answer_request: AnswerRequest, factuality: bool = False):
    """
    Generate an answer based on the provided answer request and factuality flag.

    Parameters:
    - answer_request: The request object containing the details for answer generation.
    - factuality: A boolean flag indicating whether to include factuality check. Defaults to False.

    Returns:
    - dict: A dictionary containing the generated answer information.
    """

    # Make sure the table exists.
    tables = db.table_names()
    if answer_request.retrieval.table not in tables:
       raise HTTPException(status_code=404, detail=INVALID_ANSWER_TABLE_ERROR)
    
    # Make sure the number of chunks isn't above 5.
    if answer_request.retrieval.chunk_count > maximum_chunk_count:
       raise HTTPException(status_code=400, detail=INVALID_CHUNK_COUNT_ERROR)

    # Get the answer.
    return_info = rag_answer(answer_request)

    # Get the factuality.
    if factuality:
       fact_score = client.factuality.check(
            reference=return_info['injected_doc'],
            text=return_info['answer']
        )
       return_info["factuality"] = fact_score['checks'][0]['score']

    return return_info


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, port=8000, host="0.0.0.0")
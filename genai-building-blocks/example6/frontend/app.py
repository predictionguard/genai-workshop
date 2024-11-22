import time
import os
import json
from pydantic import BaseModel
from typing import List, Optional

import requests
from predictionguard import PredictionGuard
import streamlit as st

client = PredictionGuard()


#---------------------#
# Streamlit config    #
#---------------------#

#st.set_page_config(layout="wide")

# Hide the hamburger menu
hide_streamlit_style = """
<style>
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)


#----------------------------#
#   Streaming setup          #
#----------------------------#

def stream_tokens(model, messages):
    for sse in client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=2000,
        stream=True
    ):
        yield sse["data"]["choices"][0]["delta"]["content"]

def gen_tokens(model, messages):
    result = client.chat.completions.create(
        model=model,
        messages=messages,
        max_tokens=1000
    )
    return result['choices'][0]['message']['content']

def stream_data(input):
    for word in input.split(" "):
        yield word + " "
        time.sleep(0.02)


#--------------------------#
# RAG setup                #
#--------------------------#

class RetrievalRequest(BaseModel):
    table: str
    query: str
    hyde: Optional[bool] = False
    chunk_count: Optional[int] = 1
    hybrid: Optional[bool] = False
    rerank: Optional[bool] = False


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

    print("REQUEST:", answer_request.dict())

    payload = json.dumps(answer_request.dict())
    headers = {'Content-Type': 'application/json'}

    response = requests.request("POST", url, headers=headers, data=payload)
    return response.json()


#----------------------------#
#   Response Classification  #
#----------------------------#

clf_system = """Given a question, classify the question into one of the following classes: "on_topic" or "other".

Respond with the class "on_topic" if the question is directly related to {topics}.

Otherwise respond with the class "other". 

Only respond with "on_topic" or "other" and no other text."""

clf_template = """Question: "{query}"

Class: """


def classify_input(text, topics):

    response = client.chat.completions.create(
        model="Hermes-3-Llama-3.1-8B",
        messages=[
            {"role": "system", "content": clf_system.format(topics=topics)},
            {"role": "user", "content": clf_template.format(query=text)}
        ],
        max_tokens=20,
        temperature=0.1
    )
    print("Class:", response["choices"][0]["message"]["content"])
    
    if "topic" in response["choices"][0]["message"]["content"]:
        return "on_topic"
    else:
        return "other"


#--------------------------#
# Streamlit sidebar        #
#--------------------------#

st.sidebar.markdown(
    "This chat interface uses [Prediction Guard](https://www.predictionguard.com) to answer questions based on uploaded documents in a Vector DB."
)

topics = st.sidebar.text_input(
    "Enter the topic(s) of the documents you have uploaded:",
    value="General reference information"
)


#--------------------------#
# Streamlit app            #
#--------------------------#

if "messages" not in st.session_state:
    st.session_state.messages = []

# Initialize the system prompt at the start of the session if not already present
if not st.session_state.messages or (st.session_state.messages and st.session_state.messages[0].get("role") != "system"):
    st.session_state.messages.insert(0, {"role": "system", "content": "You are a helpful assistant."})

for message in [msg for msg in st.session_state.messages if msg["role"] != "system"]:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask a question..."):

    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response in chat message container
    with st.chat_message("assistant"):

        if classify_input(prompt, topics) == "other":
            response = st.write_stream(stream_tokens(
                    "Hermes-3-Llama-3.1-8B", 
                    st.session_state.messages
                ))
            
        else:
            answer_request = AnswerRequest(
                retrieval=RetrievalRequest(
                    table="wiki",
                    query=prompt,
                    hyde=True,
                    chunk_count=1,
                    hybrid=True
                ),
                llm=LLMConfig(
                    temperature=0.1
                )
            )
            rag_response = rag_answer(answer_request)
            response = st.write_stream(stream_data(rag_response["answer"]))
        
    st.session_state.messages.append({"role": "assistant", "content": response})
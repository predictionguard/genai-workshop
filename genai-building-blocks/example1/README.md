# Streamlit frontend

This is a Chat UI, written in Python (with Streamlit).

## Running locally via Python

Install the dependencies: 

```
$ pip install -r requirements.txt
```

Run the API:

```
$ RAG_API_URL=<the url to your backend> RAG_API_TABLE=<the table name where your docs are loaded> streamlit run chat.py
```

## Running via Docker

Build the docker image:

```
$ docker build -t ragfrontend
```

Run the docker image:

```
$ docker run -it -p 8501:8501 -e RAG_API_URL=<the url to your backend> -e RAG_API_TABLE=<the table name where your docs are loaded> ragfrontend
```
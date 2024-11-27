# Retrieval Augmented Generation Backend

This is a REST API, written in Python (with FastAPI) that allows a user to:

- Upload documents to a vector database
- Update documents in the vector database
- Search the vector database via a natural language query
- Get a RAG LLM answer
- Create tables in the vector database
- Delete tables in the vector database

## Running locally via Python

Install the dependencies: 

```
$ pip install -r requirements.txt
```

Run the API:

```
$ PREDICTIONGUARD_TOKEN=<your PG access token> python main.py
```

## Running via Docker

Build the docker image:

```
$ docker build -t ragapi
```

Run the docker image:

```
$ docker run -it -p 8000:8000 -e PREDICTIONGUARD_TOKEN=<your PG access token> ragapi
```

### Using the API

Once you have deployed the API, visit `<api url>/docs` in a browser to see the swagger documentation. A basic workflow might be:

Step 1: Load docs:

```
$ curl --location '<api url>/docs' \
--header 'Content-Type: application/json' \
--data '{
    "table": "wiki",
    "docs": [
        {
            "doc": "Linux (/ˈlɪnʊks/ LIN-uuks)[11] is a family of open-source Unix-like operating systems based on the Linux kernel,[12] an operating system kernel first released on September 17, 1991, by Linus Torvalds.[13][14][15] Linux is typically packaged as a Linux distribution (distro), which includes the kernel and supporting system software and libraries, many of which are provided by the GNU Project.",
            "metadata": "from linux wiki"
        },
        {
            "doc": "Martin Luther OSA (/ˈluːθər/;[1] German: [ˈmaʁtiːn ˈlʊtɐ] ⓘ; 10 November 1483[2]– 18 February 1546) was a German priest, theologian, author, hymnwriter, professor, and Augustinian friar.[3] He was the seminal figure of the Protestant Reformation, and his theological beliefs form the basis of Lutheranism. ",
            "metadata": "luther wiki"
        },
        {
            "doc": "A large language model (LLM) is a language model notable for its ability to achieve general-purpose language generation and understanding. LLMs acquire these abilities by learning statistical relationships from text documents during a computationally intensive self-supervised and semi-supervised training process.[1] LLMs are artificial neural networks, the largest and most capable of which are built with a transformer-based architecture. Some recent implementations are based on other architectures, such as recurrent neural network variants and Mamba (a state space model).[2][3][4]",
            "metadata": "AI wiki"
        }
    ]
}'
```

Step 2: Query

```
curl --location '<api url>/answers' \
--header 'Content-Type: application/json' \
--data '{
    "query": "Who was Martin Luther?",
    "table": "wiki"
}'
```

This returns:

```
{
    "answer": "Martin Luther was a German priest, theologian, author, hymnwriter, professor, and Augustinian friar who played a significant role in the Protestant Reformation and is considered the founder of Lutheranism.",
    "injected_doc": "Martin Luther OSA (/ˈluːθər/;[1] German: [ˈmaʁtiːn ˈlʊtɐ] ⓘ; 10 November 1483[2]– 18 February 1546) was a German priest, theologian, author, hymnwriter, professor, and Augustinian friar.[3] He was the seminal figure of the Protestant Reformation, and his theological beliefs form the basis of Lutheranism. ",
    "injected_doc_id": "79f76376-44d0-4aa4-adda-cd0bed450145",
    "_distance": 0.7518587112426758
}
```

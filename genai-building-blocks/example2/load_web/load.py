import requests
import json
import os
import urllib
import sys

import html2text
from langchain.text_splitter import CharacterTextSplitter
import sys
import os
import yaml
import munch


ymlcfg = yaml.safe_load(open(os.path.join(sys.path[0],  'load.yml')))
cfg = munch.munchify(ymlcfg)


def load_docs(docs, table):

    url = os.environ.get("RAG_API_URL")
    if url is None:
        url = cfg.RAG_API_URL + '/docs'

    payload = json.dumps({
        "table": table,
        "docs": docs
    })
    headers = {
    'Content-Type': 'application/json'
    }

    response = requests.request("POST", url, headers=headers, data=payload)
    if response.status_code != 200:
        raise Exception(response)

    return

def load_webpage(url):

    # Let's get the html off of a website.
    fp = urllib.request.urlopen(url)
    mybytes = fp.read()
    html = mybytes.decode("utf8")
    fp.close()
    
    # And convert it to text.
    h = html2text.HTML2Text()
    h.ignore_links = True
    text = h.handle(html)
    
    # Chunk the text into smaller pieces for injection into LLM prompts.
    text_splitter = CharacterTextSplitter(
        chunk_size=700, 
        chunk_overlap=50,
        separator=" ",
    )
    docs = text_splitter.split_text(text)
    
    # A bit of cleanup.
    docs = [x.replace('#', '-') for x in docs]

    docs_out = []
    for d in docs:
        docs_out.append({'doc': d, 'metadata': url})
        
    return docs_out

def load(url, table):
    docs = load_webpage(url)
    batch_size = 10
    for i in range(0, len(docs), batch_size):
        load_docs(docs[i:i+batch_size], table)
    return

if __name__ == "__main__":
    url = os.environ.get("LOAD_URL")
    if url is None:
        url = cfg.LOAD_URL
    
    table = os.environ.get("LOAD_TABLE")
    if table is None:
        table = cfg.LOAD_TABLE

    load(url, table)
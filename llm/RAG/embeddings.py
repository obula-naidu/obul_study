#  This just shows how to get the embedding vector for a given text using the nomic-embed-text model.
#  The vector can then be used for various applications such as similarity search, clustering, etc.

import requests

resp = requests.post(
    "http://localhost:11434/api/embeddings",
    json={
        "model": "nomic-embed-text",
        "prompt": "FastAPI is a Python framework but Flask is also a Python framework"
    }
)

vector = resp.json()["embedding"]
print(len(vector))

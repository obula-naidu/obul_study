# import requests

# URL = "http://localhost:11434/api/chat"

# payload = {
#     "model": "llama3.2:latest",
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "What is fastapi?"}
#     ],
#     "stream": False
# }

# response = requests.post(URL, json=payload)
# data = response.json()

# print(data["message"]["content"])

import requests
import json

URL = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3.2",
    "messages": [
        {"role": "system", "content": "You are a backend expert"},
        {"role": "user", "content": "Explain FastAPI vs Flask"}
    ],
    "stream": True
}

with requests.post(URL, json=payload, stream=True) as r:
    for line in r.iter_lines():
        if line:
            chunk = json.loads(line.decode())
            if "message" in chunk:
                print(chunk["message"]["content"], end="", flush=True)


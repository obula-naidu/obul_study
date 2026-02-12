Different endpoints = different response keys

/api/chat

{
  "message": {
    "role": "assistant",
    "content": "answer text"
  }
}


/api/generate

{
  "response": "answer text"
}


/api/embeddings

{
  "embedding": [0.123, -0.456, ...]
}




STEP 1 — MINIMAL LLM CALL (ABSOLUTE BASICS)

👉 Concept: “LLM is just an HTTP API”

# llm_basics.py
import requests

URL = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3",
    "messages": [
        {"role": "user", "content": "What is REST?"}
    ],
    "stream": False
}

response = requests.post(URL, json=payload)
data = response.json()

print(data["message"]["content"])

What you learned

LLM = HTTP POST

No memory

No system message yet

Run:

python llm_basics.py

STEP 2 — SYSTEM vs USER (BEHAVIOR CONTROL)

👉 Concept: System message controls HOW the model behaves

Modify code:

payload = {
    "model": "llama3",
    "messages": [
        {"role": "system", "content": "You are a senior backend engineer. Answer concisely."},
        {"role": "user", "content": "What is REST?"}
    ],
    "stream": False
}


Run again.

What changed?

Same question, different tone & depth.

✅ This is prompt engineering at the correct level.

STEP 3 — TEMPERATURE (DETERMINISM vs CREATIVITY)

👉 Concept: Temperature controls randomness

Add temperature:

payload = {
    "model": "llama3",
    "temperature": 0.1,
    "messages": [
        {"role": "system", "content": "You are a backend engineer."},
        {"role": "user", "content": "Explain REST APIs"}
    ],
    "stream": False
}


Run twice → output is very similar.

Now change:

"temperature": 0.9


Run again.

What you learned

Low temp → stable (good for logs, analysis)

High temp → variation (bad for backend systems)

STEP 4 — STATELESS NATURE (VERY IMPORTANT)

👉 Concept: LLM does NOT remember previous calls

Try this:

payload = {
    "model": "llama3",
    "messages": [
        {"role": "user", "content": "My name is Obul"},
        {"role": "user", "content": "What is my name?"}
    ],
    "stream": False
}


Now try removing first message.

What you learned

Memory exists only if you send it

Chat history is YOUR responsibility

STEP 5 — STRUCTURED OUTPUT (BACKEND-GRADE SKILL)

👉 Concept: LLMs must return machine-readable output

payload = {
    "model": "llama3",
    "temperature": 0.2,
    "messages": [
        {
            "role": "system",
            "content": (
                "You are a diagnostic assistant.\n"
                "Respond ONLY in valid JSON with keys:\n"
                "root_cause, severity, recommendation"
            )
        },
        {
            "role": "user",
            "content": "Service is slow and CPU usage is high"
        }
    ],
    "stream": False
}


Output:

{
  "root_cause": "...",
  "severity": "...",
  "recommendation": "..."
}


🔥 This is RAIN-style output.

STEP 6 — PARSE LLM OUTPUT (REAL BACKEND USAGE)

👉 Concept: LLM output is input to your program

import json

result = data["message"]["content"]
parsed = json.loads(result)

print("Root cause:", parsed["root_cause"])


Now the LLM is a component, not a chatbot.

STEP 7 — STREAMING (HOW CHAT UIs WORK)

👉 Concept: Token-by-token output

Replace everything with:

import requests
import json

URL = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3",
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

What you learned

Streaming = multiple JSON chunks

Required for real chat apps

Harder than normal HTTP

STEP 8 — REUSABLE FUNCTION (ENGINEER MOVE)

👉 Concept: Abstract LLM behind a function

def call_llm(system, user, temperature=0.2):
    payload = {
        "model": "llama3",
        "temperature": temperature,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
        "stream": False
    }

    r = requests.post(URL, json=payload)
    return r.json()["message"]["content"]


Now you can call:

print(call_llm("You are a backend expert", "Explain REST"))

# about ollama client usage, chat, generate, embeddings

from client import OllamaClient

client = OllamaClient()
MODEL = "llama3.2:latest"

#  Chat
chat_response = client.chat(
    model=MODEL,
    messages=[
        {"role": "system", "content": "You are a backend expert"},
        {"role": "user", "content": "What is FastAPI?"}
    ]
)
print("CHAT:", chat_response)

#  Generate
gen_response = client.generate(
    model=MODEL,
    prompt="Explain FastAPI in one paragraph"
)
print("GENERATE:", gen_response)

# Embeddings
embedding = client.embeddings(
    model=MODEL,
    text="FastAPI is a Python web framework"
)
print("EMBEDDING length:", len(embedding))

import requests

class OllamaClient:
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def chat(self, model, messages, stream=False):
        url = f"{self.base_url}/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "stream": stream
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return data["message"]["content"]

    def generate(self, model, prompt, stream=False):
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": stream
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return data["response"]

    def embeddings(self, model, text):
        url = f"{self.base_url}/api/embeddings"
        payload = {
            "model": model,
            "prompt": text
        }

        response = requests.post(url, json=payload)
        response.raise_for_status()

        data = response.json()
        return data["embedding"]

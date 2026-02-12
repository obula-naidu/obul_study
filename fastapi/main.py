from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def hello():
    return {"message": "Hello, World!"}

@app.get("/hello/{name}")
def hello(name):
    return f"hello {name}"
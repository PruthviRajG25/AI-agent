import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()
client = OpenAI(
    base_url=os.getenv("BASE_URL", "http://127.0.0.1:11434/v1"),
    api_key=os.getenv("API_KEY", "dummy"),
)

response = client.embeddings.create(
    model=os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest"),
    input=["Hello, my dog is cute","Ram is good boy","I love programming in Python","The weather is nice today"],
)

for i, embedding in enumerate(response.data):
    print("length: ",len(embedding.embedding))
    print(embedding.embedding[:3])  
import os
from pathlib import Path

import chromadb
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

client = OpenAI(
    base_url=os.getenv("BASE_URL", "http://127.0.0.1:11434/v1"),
    api_key=os.getenv("API_KEY", "dummy"),
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")

def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
    )
    return response.data[0].embedding


chunks = {
    "c0": "Leave Policy - Maternity Leave\nMaternity leave is 26 weeks of paid leave for the first two children.",
    "c1": "Medical Policy - Maternity Cover\nHospitalization for childbirth is covered up to 75,000 rupees.",
    "c2": "Travel Expense - Reimbursement\nSubmit expense claims within 30 days of the trip. Approved claims are reimbursed within 7 working days.",
    "c3": "Tech Docs - Deployment Windows\nProduction deployments are allowed on Tuesdays and Thursdays between 11 AM and 4 PM.",
}

ids = list(chunks.keys())
documents = list(chunks.values())

embeddings = []
for s in documents:
    embedding = get_embedding(s)
    embeddings.append(embedding)


chroma = chromadb.PersistentClient(path="./chromadbStorage")

zorvex_collection = chroma.get_or_create_collection(
    name="zorvex_technology",
    configuration={
        "hnsw": {
            "space": "cosine",
        }
    }
)

zorvex_collection.upsert( #It will update if an Id is existing, otherwise it will insert a new record
    ids=ids,
    documents=documents,
    embeddings=embeddings,
)




query = "what is leave policy?"
query_embedding = get_embedding(query)

results = zorvex_collection.query(
    query_embeddings=[query_embedding],
    n_results=1,
)

print(results["documents"][0])
print(results["distances"][0])






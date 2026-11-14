import os
import numpy
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv() #by default looks for .env file

client = OpenAI(
    base_url=os.getenv("BASE_URL", "http://127.0.0.1:11434/v1"),
    api_key=os.getenv("API_KEY", "dummy")
)

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text:latest")
KB_FOLDER = os.getenv("KB_FOLDER", "knowledge_base")


#Get the embedding of a text from Embedding model
def get_embedding(text):
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text
    )
    return response.data[0].embedding

#Get the cosine similarity between two vectors
def cosine_similarity(A, B):
    dot_prod = numpy.dot(A, B)
    norm_a = numpy.linalg.norm(A)
    norm_b = numpy.linalg.norm(B)
    return dot_prod / (norm_a * norm_b)


#Load the documents
def load_documents(folder=KB_FOLDER):
    documents = []
    
    for filename in sorted(os.listdir(folder)):
        if not filename.endswith(".txt"):
            continue

        path = os.path.join(folder, filename)
        with open(path, "r", encoding="utf-8") as f:
            documents.append(
                {
                    "source": filename,
                    "text":f.read(),
                }
            )
        
    return documents

#Add embeddings in the documents
def add_embeddings(documents):
    for document in documents:
        vector = get_embedding(document["text"])
        document["vector"] = vector
    
    return documents


#Search method to search the top_k documents for a user_query
def search(query, documents, top_k=3):
    query_vector = get_embedding(query)

    results = []
    for document in documents:
        score = cosine_similarity(query_vector, document["vector"])
        results.append(
            {
                "score" : float(score),
                "source": document["source"],
                "text": document["text"],
                "size": len(document["text"])
            }
        )

    results.sort(key=lambda rec: rec["score"], reverse=True)
    return results[:top_k]
    

#testing method
def test(user_query, documents, top_k=3):
    print(f"\nUser Query: {user_query}\n")

    results = search(user_query, documents, top_k)

    for r in results:
        print(f" {r['score']:.2f} - {r['source']} - {r['size']} characters)")
    
    

#Let's start testing
documents = load_documents()
documents = add_embeddings(documents)

print(f"We have embedded {len(documents)} documents")

user_query = "hello  ?"
test(user_query, documents, 3)

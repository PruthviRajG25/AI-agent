from pathlib import Path

import chromadb


KNOWLEDGE_BASE_DIR = Path(__file__).parent / "knowledge_base"


def main() -> None:
    documents = sorted(KNOWLEDGE_BASE_DIR.glob("*.txt"))
    if not documents:
        raise FileNotFoundError(
            f"No .txt documents found in {KNOWLEDGE_BASE_DIR}"
        )

    client = chromadb.Client()
    collection = client.create_collection(name="knowledge_base")
    collection.add(
        ids=[document.stem for document in documents],
        documents=[
            document.read_text(encoding="utf-8") for document in documents
        ],
        metadatas=[{"source": document.name} for document in documents],
    )

    result = collection.query(
        query_texts=["How many sick leaves are allowed?"],
        n_results=1,
    )

    print(f"Loaded {len(documents)} documents from {KNOWLEDGE_BASE_DIR}")
    for source, document, distance in zip(
        result["metadatas"][0],
        result["documents"][0],
        result["distances"][0],
    ):
       print(
            f"Source: {source['source']}, Distance: {distance:.2f}\n"
            f"Document: {document}\n"
        )


if __name__ == "__main__":
    main()
import os
from pathlib import Path


from anthropic import Anthropic
from dotenv import load_dotenv


def load_documents():
    documents_folder = Path("documents")

    files = list(documents_folder.glob("*.txt"))

    documents = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append({"text": text, "filename": file.name})

    return documents


documents = load_documents()


def chunk_document(text):
    lines = text.strip().split("\n")

    lines = [line.strip() for line in lines if line.strip()]

    chunks = []
    current_chunk = ""

    for line in lines:

        if not line.endswith("."):

            if current_chunk:
                chunks.append(current_chunk)

            current_chunk = line

        else:
            current_chunk += "\n" + line

    if current_chunk:
        chunks.append(current_chunk)

    chunks = [chunk for chunk in chunks if len(chunk.strip().splitlines()) > 1]
    return chunks


def prepare_chunks(documents):
    chunk_data = []

    for document in documents:
        chunks = chunk_document(document["text"])
        filename = document["filename"]

        for chunk in chunks:

            if filename == "employee_handbook.txt":
                category = "HR"
            else:
                category = "Company Policy"

            chunk_data.append(
                {"text": chunk, "document": filename, "category": category}
            )
    return chunk_data


chunk_data = prepare_chunks(documents)


def create_vector_database(chunk_data):

    import chromadb

    chroma_client = chromadb.Client()

    collection = chroma_client.create_collection(name="support_documents_v2")

    ids = [str(i) for i in range(len(chunk_data))]

    documents = [item["text"] for item in chunk_data]
    metadata = [
        {"document": item["document"], "category": item["category"]}
        for item in chunk_data
    ]
    collection.add(ids=ids, documents=documents, metadatas=metadata)

    return collection


collection = create_vector_database(chunk_data)


def retrieve_documents(
    collection, question, threshold=0.7, document_name=None, category=None
):
    where_filter = {}

    if document_name:
        where_filter["document"] = document_name

    if category:
        where_filter["category"] = category

    results = collection.query(
        query_texts=[question],
        n_results=4,
        where=where_filter if where_filter else None,
    )

    distances = results["distances"][0]
    documents = results["documents"][0]
    metadata = results["metadatas"][0]

    # for document, distance, meta in zip(documents, distances, metadata):
    #     print("Distance:", distance)
    #     print("Document:", meta["document"])
    #     print()
    relevant_results = []

    for document, distance, meta in zip(documents, distances, metadata):
        if distance < threshold:
            relevant_results.append(
                {"text": document, "distance": distance, "metadata": meta}
            )

    return relevant_results


def generate_answer(question, relevant_results):
    context = "\n\n".join(result["text"] for result in relevant_results)
    load_dotenv()

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=500,
        messages=[
            {
                "role": "user",
                "content": f"""
        Answer the question using only the information provided below.
        Information: {context}
        Question:
        {question} """,
            }
        ],
    )
    source = relevant_results[0]["metadata"]["document"]
    return response.content[0].text, source


def detect_category(question):
    load_dotenv()

    client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=50,
        messages=[
            {
                "role": "user",
                "content": f"""
            Classify the question into exactly one of these categories:

                            - HR
                - Company Policy
                - None

                Return only the category name.

                Question:
                {question}
                """,
            }
        ],
    )
    return response.content[0].text.strip()

def main ():
    question = input("Ask a question:")
    category = detect_category(question)
    print("Detected category", category)
    relevant_results = retrieve_documents(
        collection, question, category=category if category != "None" else None
    )
    print("Retrieved", len(relevant_results), "document(s)")

    if not relevant_results:
        print("I don't have enough information to answer this question.")
    else:
        answer, source = generate_answer(question, relevant_results)

        print("\nClaude", answer)
        print("Source:", source)

if __name__ == "__main__":
    main()

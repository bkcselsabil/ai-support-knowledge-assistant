import os
from pathlib import Path


from anthropic import Anthropic
from dotenv import load_dotenv


def initialize_rag():
    # Load and chunk local documents
    local_documents = load_documents()
    local_chunks = prepare_chunks(local_documents)

    # Fetch and filter Confluence pages
    from confluence_ingestion import (
        fetch_all_confluence_pages,
        filter_confluence_pages,
        prepare_confluence_chunks,
        combine_chunks,
    )

    confluence_pages = fetch_all_confluence_pages("ASKB")
    filtered_pages = filter_confluence_pages(confluence_pages)
    confluence_chunks = prepare_confluence_chunks(filtered_pages)

    # Combine local and Confluence chunks
    combined_chunks = combine_chunks(local_chunks, confluence_chunks)

    # Create the collection
    collection = create_vector_database(combined_chunks)
    return collection


def load_documents():
    documents_folder = Path("documents")

    files = list(documents_folder.glob("*.txt"))

    documents = []

    for file in files:
        with open(file, "r", encoding="utf-8") as f:
            text = f.read()

        documents.append({"text": text, "filename": file.name})

    return documents


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



def create_vector_database(chunk_data, collection_name="support_documents_v2",):
    import chromadb

    chroma_client = chromadb.Client()

    try:
        chroma_client.delete_collection(name=collection_name)
    except Exception:
        pass

    collection = chroma_client.create_collection(name=collection_name)
    ids = [str(i) for i in range(len(chunk_data))]
    documents = [item["text"] for item in chunk_data]

    metadata = []
    for item in chunk_data:
        item_metadata = {
            "document": item["document"],
            "category": item["category"],
            "source": item.get("source", "local"),

        }

        for field in ["source", "title", "page_id", "space_key", "url"]:
            if item.get(field) is not None:
                item_metadata[field] = item[field]

        metadata.append(item_metadata)

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadata,
    )

    return collection



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


collection = initialize_rag()


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
        If the information does not answer the question, say so.
        Information: {context}
        Question:
        {question}
        """,
            }
        ],
    )

    sources = []
    seen_sources = set()

    for result in relevant_results:
        metadata = result["metadata"]
        source_key = metadata.get("url") or metadata["document"]

        if source_key not in seen_sources:
            seen_sources.add(source_key)
            sources.append({
                "title": metadata.get("title", metadata["document"]),
                "document": metadata["document"],
                "url": metadata.get("url"),
                "source": metadata.get("source", "local"),
            })

    if all(source["source"] == "local" for source in sources):
        return response.content[0].text, sources[0]["document"]

    return response.content[0].text, sources



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

        answer, sources = generate_answer(question, relevant_results)

        print("\nClaude:", answer)
        print("Source:" if isinstance(sources, str) else "\nSources:")

        if isinstance(sources, str):
            # Backward compatibility with existing callers and tests
            print("Source:", sources)
        else:
            for source in sources:
                print("-", source["title"])

                if source["url"]:
                    print("  URL:", source["url"])
                else:
                    print("  Document:", source["document"])


if __name__ == "__main__":
    main()

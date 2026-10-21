
from rag import create_vector_database

sample_chunks = [
    {
        "text": "Employees receive 30 vacation days per year.",
        "document": "employee_handbook.txt",
        "category": "HR",
    },
    {
        "text": "Employees may work remotely three days per week.",
        "document": "confluence_720897.txt",
        "category": "Company Policy",
        "source": "confluence",
        "title": "Remote Work Policy",
        "page_id": "720897",
        "space_key": "ASKB",
        "url": "https://selsabilprojob.atlassian.net/wiki/spaces/ASKB/pages/720897",
    },
]
collection = create_vector_database(
    sample_chunks,
    collection_name="test_chroma_metadata",
)
results = collection.get(
    include=["documents", "metadatas"]
)

for document, metadata in zip(
    results["documents"], results["metadatas"]
):
    print("\nDocument:", document)
    print("Metadata:", metadata)

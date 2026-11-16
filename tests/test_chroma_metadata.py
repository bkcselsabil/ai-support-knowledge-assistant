
from rag import create_vector_database
def test_chroma_metadata():
    sample_chunks = [
        {
            "text": "Employees receive 30 vacation days per year.",
            "document": "employee_handbook.txt",
            "category": "HR",
            "tenant_id": "local-dev",
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
            "tenant_id": "local-dev",
        },
    ]
    import chromadb
    from unittest.mock import patch

    test_client = chromadb.EphemeralClient()

    with patch("chromadb.PersistentClient", return_value=test_client):
        collection = create_vector_database(
            sample_chunks,
            collection_name="test_chroma_metadata",
        )
    results = collection.get(
        include=["documents", "metadatas"]
    )
    assert len(results["ids"]) == 2

    assert all(
        metadata["tenant_id"] == "local-dev"
        for metadata in results["metadatas"]
    )
    for document, metadata in zip(
        results["documents"], results["metadatas"]
    ):
        print("\nDocument:", document)
        print("Metadata:", metadata)

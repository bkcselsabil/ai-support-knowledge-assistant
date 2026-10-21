
from rag import load_documents, prepare_chunks, create_vector_database
from confluence_ingestion import (
    fetch_all_confluence_pages,
    filter_confluence_pages,
    prepare_confluence_chunks,
    combine_chunks,
)


# Step 1: Load local documents
local_documents = load_documents()
local_chunks = prepare_chunks(local_documents)

# Step 2: Fetch and filter Confluence pages
all_confluence_pages = fetch_all_confluence_pages("ASKB")
filtered_pages = filter_confluence_pages(all_confluence_pages)

# Step 3: Convert Confluence pages into chunks
confluence_chunks = prepare_confluence_chunks(filtered_pages)

# Step 4: Combine both sources
combined_chunks = combine_chunks(local_chunks, confluence_chunks)

# Step 5: Verify the result
print("Local documents:", len(local_documents))
print("Local chunks:", len(local_chunks))
print("Confluence pages discovered:", len(all_confluence_pages))
print("Confluence pages selected:", len(filtered_pages))
print("Confluence chunks:", len(confluence_chunks))
print("Total combined chunks:", len(combined_chunks))

print("\nConfluence metadata check:")
for chunk in combined_chunks:
    if chunk.get("source") == "confluence":
        print("Title:", chunk["title"])
        print("Page ID:", chunk["page_id"])
        print("URL:", chunk["url"])
        break

print("\nFirst local chunk fields:")
if local_chunks:
    print(list(local_chunks[0].keys()))

print("\nFirst Confluence chunk fields:")
if confluence_chunks:
    print(list(confluence_chunks[0].keys()))


print("\nCreating a test collection with all chunks...")

collection = create_vector_database(
    combined_chunks,
    collection_name="test_combined_documents",
)

stored = collection.get(
    include=["documents", "metadatas"]
)

print("Chunks stored in ChromaDB:", len(stored["documents"]))

confluence_items = [
    metadata
    for metadata in stored["metadatas"]
    if metadata.get("source") == "confluence"
]

print("Confluence chunks stored:", len(confluence_items))

if confluence_items:
    print("Example stored Confluence metadata:", confluence_items[0])

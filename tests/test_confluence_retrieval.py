
from rag import collection, retrieve_documents

question = "How many days per week can employees work remotely?"

results = retrieve_documents(
    collection,
    question,
    threshold=0.7,
)

print("Question:", question)
print("Retrieved chunks:", len(results))

for result in results:
    metadata = result["metadata"]

    print("\nText:", result["text"])
    print("Distance:", result["distance"])
    print("Source:", metadata.get("source", "local"))
    print("Title:", metadata.get("title", metadata.get("document")))
    print("URL:", metadata.get("url", "No Confluence URL"))

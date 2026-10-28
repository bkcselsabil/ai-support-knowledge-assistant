import os
import requests
from dotenv import load_dotenv
from requests.auth import HTTPBasicAuth
from bs4 import BeautifulSoup
from app.documents import SourceDocument

load_dotenv()


def get_confluence_credentials():
    base_url = os.getenv("CONFLUENCE_BASE_URL")
    email = os.getenv("CONFLUENCE_EMAIL")
    api_token = os.getenv("CONFLUENCE_API_TOKEN")

    if not all([base_url, email, api_token]):
        raise ValueError("Missing Confluence credentials in .env")

    return base_url, email, api_token


def fetch_confluence_page(page_id):
    base_url, email, api_token = get_confluence_credentials()

    url = f"{base_url}/wiki/rest/api/content/{page_id}"

    response = requests.get(
        url,
        auth=HTTPBasicAuth(email, api_token),
        headers={"Accept": "application/json"},
        params={"expand": "space,version,body.storage"},
        timeout=20,
    )

    response.raise_for_status()

    page = response.json()

    html_content = page.get("body", {}).get("storage", {}).get("value", "")

    soup = BeautifulSoup(html_content, "html.parser")
    plain_text = soup.get_text(separator="\n", strip=True)

    return {
        "text": plain_text,
        "title": page.get("title", ""),
        "page_id": str(page.get("id", page_id)),
        "space_key": page.get("space", {}).get("key", ""),
        "source": "confluence",
        "url": f"{base_url}/wiki/spaces/{page.get('space', {}).get('key', '')}/pages/{page.get('id', page_id)}",
    }


def fetch_confluence_pages(page_ids):
    documents = []

    for page_id in page_ids:
        page = fetch_confluence_page(page_id)

        documents.append(
            {
                "text": page["text"],
                "filename": f"confluence_{page['page_id']}.txt",
                "title": page["title"],
                "page_id": page["page_id"],
                "space_key": page["space_key"],
                "source": page["source"],
                "url": page["url"],
            }
        )

    return documents



def convert_confluence_documents(documents, tenant_id):
    """Convert already-fetched Confluence pages into SourceDocument objects."""

    source_documents = []

    for document in documents:
        page_id = str(document["page_id"])

        source_documents.append(
            SourceDocument(
                tenant_id=tenant_id,
                source=document.get("source", "confluence"),
                document_id=page_id,
                title=document.get("title", ""),
                text=document.get("text", ""),
                url=document.get("url"),
                metadata={
                    "filename": document.get(
                        "filename", f"confluence_{page_id}.txt"
                    ),
                    "space_key": document.get("space_key", ""),
                },
            )
        )

    return source_documents



def prepare_confluence_chunks(documents):
    from rag import chunk_document

    chunk_data = []

    for document in documents:
        if isinstance(document, SourceDocument):
            text = document.text
            title = document.title
            page_id = document.document_id
            source = document.source
            url = document.url
            tenant_id = document.tenant_id
            space_key = document.metadata.get("space_key", "")
            filename = document.metadata.get(
                "filename", f"confluence_{page_id}.txt"
            )
        else:
            text = document["text"]
            title = document["title"]
            page_id = document["page_id"]
            source = document["source"]
            url = document["url"]
            tenant_id = document.get("tenant_id", "local-dev")
            space_key = document["space_key"]
            filename = document.get(
                "filename", f"confluence_{page_id}.txt"
            )

        chunks = chunk_document(text)

        for chunk in chunks:
            chunk_data.append(
                {
                    "text": chunk,
                    "document": filename,
                    "category": (
                        "HR"
                        if "vacation" in title.lower()
                        else "Company Policy"
                    ),
                    "source": source,
                    "title": title,
                    "page_id": page_id,
                    "space_key": space_key,
                    "url": url,
                    "tenant_id": tenant_id,
                }
            )

    return chunk_data

def combine_chunks(local_chunks, confluence_chunks):
    combined_chunks = local_chunks + confluence_chunks
    return combined_chunks


def list_confluence_pages(space_key, limit=50):
    base_url, email, api_token = get_confluence_credentials()

    url = f"{base_url}/wiki/rest/api/content"

    pages = []
    start = 0

    while True:
        response = requests.get(
            url,
            auth=HTTPBasicAuth(email, api_token),
            headers={"Accept": "application/json"},
            params={
                "spaceKey": space_key,
                "type": "page",
                "limit": limit,
                "start": start,
            },
            timeout=20,
        )

        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        for page in results:
            pages.append(
                {
                    "page_id": str(page["id"]),
                    "title": page["title"],
                    "space_key": space_key,
                }
            )

        if len(results) < limit:
            break

        start += len(results)

    return pages


def fetch_all_confluence_pages(space_key):
    pages = list_confluence_pages(space_key)
    documents = []

    for page in pages:
        document = fetch_confluence_page(page["page_id"])
        documents.append(document)

    return documents


def filter_confluence_pages(documents):
    excluded_titles = {"AI Support Knowledge Base"}
    filtered_documents = []

    for document in documents:
        title = document["title"]

        if title in excluded_titles:
            continue

        if title.startswith("Template -"):
            continue

        if not document["text"].strip():
            continue

        filtered_documents.append(document)

    return filtered_documents


if __name__ == "__main__":
    documents = fetch_all_confluence_pages("ASKB")
    filtered_documents = filter_confluence_pages(documents)

    chunks = prepare_confluence_chunks(filtered_documents)

    print("Pages discovered:", len(documents))
    print("Pages selected for indexing:", len(filtered_documents))
    print("Chunks created:", len(chunks))

    for chunk in chunks:
        print("\nTitle:", chunk["title"])
        print("Document:", chunk["document"])
        print("Category:", chunk["category"])
        print("Source:", chunk["source"])
        print("Page ID:", chunk["page_id"])
        print("URL:", chunk["url"])
        print("Text preview:", chunk["text"][:100])
import os

from anthropic import Anthropic
from dotenv import load_dotenv
from rag import (
    retrieve_documents,
    get_document_inventory,
    get_document_chunks,
    collection,
)
import json

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

def filter_relevant_documents(question, candidate_results):
    """
    Use Claude tool calling to identify genuinely relevant document chunks.
    """

    if not candidate_results:
        return []

    candidates = []

    for index, result in enumerate(candidate_results):
        candidates.append({
            "index": index,
            "document": result["metadata"].get("document"),
            "content": result["text"],
        })

    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=200,
        system="""
You are a strict document relevance evaluator.

Determine which document chunks contain information
that directly helps answer the user's question.

Rules:
- Keep only genuinely relevant chunks.
- Similar wording alone does not make a chunk relevant.
- Do not infer information.
- Return an empty list if no chunks are relevant.
""",
        tools=[
            {
                "name": "submit_relevance_decision",
                "description": "Submit the indices of relevant document chunks.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "relevant_indices": {
                            "type": "array",
                            "items": {"type": "integer"},
                            "description": "Indices of relevant document chunks."
                        }
                    },
                    "required": ["relevant_indices"],
                },
            }
        ],
        tool_choice={
            "type": "tool",
            "name": "submit_relevance_decision",
        },
        messages=[
            {
                "role": "user",
                "content": json.dumps({
                    "question": question,
                    "candidates": candidates,
                }),
            }
        ],
    )

    try:
        decision = next(
            block for block in response.content
            if block.type == "tool_use"
            and block.name == "submit_relevance_decision"
        )

        relevant_indices = decision.input["relevant_indices"]

        return [
            candidate_results[index]
            for index in relevant_indices
            if type(index) is int
            and 0 <= index < len(candidate_results)
        ]

    except (StopIteration, KeyError, TypeError, IndexError) as e:
        print("Relevance evaluation failed:", repr(e))
        return []

def add_numbers(a, b):
    return a + b


def multiply_numbers(a, b):
    return a * b


def subtract_numbers(a, b):
    return a - b


def divide_numbers(a, b):
    try:
        result = a / b

    except ZeroDivisionError:
        return "Cannot divide by zero."
    else:
        return result





def search_knowledge(question, tenant_id="local-dev", document_name=None):

    # If a document is specified, verify that it belongs to this tenant.
    if document_name:
        inventory = get_document_inventory(
            collection,
            tenant_id=tenant_id,
        )

        valid_documents = {
            document["document"] for document in inventory
        }

        if document_name not in valid_documents:
            return json.dumps({
                "information": "The requested document was not found.",
                "sources": [],
            })

    candidate_results = retrieve_documents(
        collection,
        question,
        document_name=document_name,
        tenant_id=tenant_id,
    )

    results = filter_relevant_documents(
        question,
        candidate_results,
    )

    if not results:
        return json.dumps({
            "information": "No relevant information found.",
            "sources": [],
        })

    information = []
    sources = []

    for result in results:
        metadata = result["metadata"]

        information.append(result["text"])

        sources.append({
            "title": metadata.get("title", metadata["document"]),
            "document": metadata["document"],
            "url": metadata.get("url"),
            "source": metadata.get("source", "local"),
        })

    return json.dumps({
        "information": "\n\n".join(information),
        "sources": sources,
    })

def get_document_overview(document_name, tenant_id="local-dev"):

    # Verify that the document exists for this tenant.
    inventory = get_document_inventory(
        collection,
        tenant_id=tenant_id,
    )

    document_info = next(
        (
            document for document in inventory
            if document["document"] == document_name
        ),
        None,
    )

    if not document_info:
        return json.dumps({
            "information": "The requested document was not found.",
            "sources": [],
        })

    # Retrieve every chunk from this document.
    chunks = get_document_chunks(
        collection,
        document_name=document_name,
        tenant_id=tenant_id,
    )

    if not chunks:
        return json.dumps({
            "information": "No content was found in this document.",
            "sources": [],
        })

    context = "\n\n".join(
        chunk["text"] for chunk in chunks
    )

    # Summarize the complete document.
    response = client.messages.create(
        model="claude-sonnet-4-5",
        max_tokens=800,
        system="""
You are a document summarization assistant.

Summarize the supplied document using only its content.

Rules:
- Cover the main topics and sections.
- Include important rules and requirements.
- Do not invent information.
- Do not add external knowledge.
- If something is unclear, say so.
""",
        messages=[
            {
                "role": "user",
                "content": f"""
Document: {document_name}

Document content:
{context}

Provide a structured overview of this document.
"""
            }
        ],
    )

    return json.dumps({
        "information": response.content[0].text,
        "sources": [{
            "title": document_info["title"],
            "document": document_info["document"],
            "url": document_info["url"],
            "source": document_info["source"],
        }],
    })

def list_knowledge_documents(tenant_id="local-dev"):
    """
    List all unique documents available in the company knowledge base.
    """

    documents = get_document_inventory(
        collection,
        tenant_id=tenant_id,
    )

    return json.dumps({
        "total_documents": len(documents),
        "documents": documents,
    })

tool_functions = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "subtract_numbers": subtract_numbers,
    "divide_numbers": divide_numbers,
    "search_knowledge": search_knowledge,
    "list_knowledge_documents": list_knowledge_documents,
    "get_document_overview": get_document_overview,
}

def execute_tool(tool_use, tenant_id="local-dev"):
    if tool_use.name not in tool_functions:
        return "Unknown tool"

    function = tool_functions[tool_use.name]
    tool_input = dict(tool_use.input)

    # The application supplies the tenant ID.
    # Never trust the AI model to choose the company.
    if tool_use.name in [
        "search_knowledge",
        "list_knowledge_documents",
        "get_document_overview",
    ]:
        tool_input["tenant_id"] = tenant_id
    try:
        result = function(**tool_input)
        return result
    except Exception as e:
        return f"Tool error: {e}"

tools = [
    {
        "name": "add_numbers",
        "description": "Add two numbers together.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "The first number."},
                "b": {"type": "number", "description": "The second number."},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "multiply_numbers",
        "description": "Multiply two numbers together.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "The first number."},
                "b": {"type": "number", "description": "The second number."},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "subtract_numbers",
        "description": "Subtract the second number from the first number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "The first number."},
                "b": {"type": "number", "description": "The second number."},
            },
            "required": ["a", "b"],
        },
    },
    {
        "name": "divide_numbers",
        "description": "Divide the first number by the second number.",
        "input_schema": {
            "type": "object",
            "properties": {
                "a": {"type": "number", "description": "The first number."},
                "b": {"type": "number", "description": "The second number."},
            },
            "required": ["a", "b"],
        },
    },
{
    "name": "search_knowledge",
    "description": "Search the company knowledge base. If the user asks about a specific document and its exact document name is known from the inventory, provide document_name to search only that document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The user's question"
            },
            "document_name": {
                "type": "string",
                "description": "The exact document filename from the knowledge base inventory. Optional."
            }
        },
        "required": ["question"],
    },
},
    {
    "name": "list_knowledge_documents",
    "description": "List all unique documents available in the company knowledge base. Use this tool when the user asks what documents, policies, or topics are available, or requests an overview of the knowledge base.",
    "input_schema": {
        "type": "object",
        "properties": {},
        "required": [],
    },
},
    {
    "name": "get_document_overview",
    "description": "Retrieve and summarize all content from a specific document. Use this when the user asks for a complete overview, summary, or all topics covered in a named document.",
    "input_schema": {
        "type": "object",
        "properties": {
            "document_name": {
                "type": "string",
                "description": "The exact document filename from the knowledge base inventory."
            }
        },
        "required": ["document_name"],
    },
},
]

def run_agent(question, history=None, tenant_id="local-dev"):
    messages = history.copy() if history else []
    messages.append({"role": "user", "content": question})
    sources = []

    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5",
            max_tokens=500,
            system="""
You are a company support assistant.

Use the available tools when you need information from the company knowledge base.

If the knowledge base does not contain the requested information,
say clearly that the information was not found.

Do not invent company policies or facts.

Use only the tool necessary to answer the user's question.

Do not call list_knowledge_documents after search_knowledge has already provided the requested information.

Do not call search_knowledge after list_knowledge_documents has already answered a question about available documents.

If search_knowledge returns relevant information, answer using that information without performing another search unless the user requests additional details.
When the user explicitly asks about a specific document:
- First identify the exact document name from the knowledge base inventory if it is not already known.
- Pass that exact document name as document_name when calling search_knowledge.
- Search only that document.
- Never invent or guess a document filename.
- If the requested document does not exist in the inventory, explain that it was not found.
When listing available documents:
- Use only the document names, titles, and categories returned by list_knowledge_documents.
- Never change or guess a document's category.
- Do not invent document descriptions or contents.
- If a title is just a filename, you may display the filename as provided.
- Preserve the exact category returned by the tool.
When the user asks for a complete overview, summary, or all topics covered in a specific document:
- Use get_document_overview.
- First identify the exact document filename from the inventory if it is not already known.
- Pass the exact filename as document_name.
- Do not use search_knowledge for a complete document overview.

When the user asks a specific question about a document:
- Use search_knowledge.
- Pass document_name when the exact filename is known.
- Do not retrieve the entire document for a specific question.
DOCUMENT BOUNDARIES:
- When the user explicitly asks about a specific document, answer using only that document.
- Do not introduce information from other documents, even if it is available in the knowledge base.
- If the requested information is not found in that document, say so clearly.
- You may mention that other documents exist, but do not summarize their contents unless the user asks.


DOCUMENT IDENTIFICATION:
- When the user refers to a document by its title or friendly name, first call list_knowledge_documents to identify its exact document identifier.
- Use the exact "document" value returned by the inventory when calling search_knowledge or get_document_overview.
- Never guess a document identifier from its title.
- Do not call search_knowledge or get_document_overview until the exact identifier is known.
- If the document is not present in the inventory, explain that it is unavailable.
""",
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_uses = [
            block
            for block in response.content
            if block.type == "tool_use"
        ]

        if not tool_uses:
            print(f"Final sources returned by run_agent: {sources}")
            return {
                "answer": response.content[0].text,
                "sources": sources,
                "messages": messages,
            }

        tool_results = []

        for tool_use in tool_uses:
            print(f"\nClaude called tool: {tool_use.name}")
            print(f"Tool input: {tool_use.input}")
            result = execute_tool(tool_use, tenant_id=tenant_id)

            print(f"Tool result: {result}")
            if tool_use.name in [
                "search_knowledge",
                "get_document_overview",
            ] and isinstance(result, str):
                try:
                    tool_data = json.loads(result)
                    print(f"Extracted sources from {tool_use.name}: {tool_data.get('sources', [])}")
                    for source in tool_data.get("sources", []):
                        if source not in sources:
                            sources.append(source)

                except json.JSONDecodeError:
                    pass

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_use.id,
                "content": str(result),
            })

        messages.append({
            "role": "user",
            "content": tool_results,
        })

def main():
    question = input("You: ")
    result = run_agent(question)

    print("Claude:", result["answer"])

    if result["sources"]:
        print("Sources:")
        for source in result["sources"]:
            print(f"- {source['title']} ({source['document']})")


if __name__ == "__main__":
    main()

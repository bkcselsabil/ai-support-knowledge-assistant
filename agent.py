import os

from anthropic import Anthropic
from dotenv import load_dotenv
from rag import retrieve_documents, collection
import json

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


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




def search_knowledge(question):
    results = retrieve_documents(collection, question, threshold=0.8)

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

tool_functions = {
    "add_numbers": add_numbers,
    "multiply_numbers": multiply_numbers,
    "subtract_numbers": subtract_numbers,
    "divide_numbers": divide_numbers,
    "search_knowledge": search_knowledge,
}


def execute_tool(tool_use):
    if tool_use.name not in tool_functions:
        return "Unknown tool"

    function = tool_functions[tool_use.name]

    try:
        result = function(**tool_use.input)
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
        "description": "Search the company knowledge base for relevant information",
        "input_schema": {
            "type": "object",
            "properties": {
                "question": {"type": "string", "description": "The user's question"}
            },
            "required": ["question"],
        },
    },
]

def run_agent(question, history=None):
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
""",
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_use = None
        for block in response.content:
            if block.type == "tool_use":
                tool_use = block
                break

        if tool_use is None:
            return {
                "answer": response.content[0].text,
                "sources": sources,
                "messages": messages,
            }

        result = execute_tool(tool_use)

        if tool_use.name == "search_knowledge" and isinstance(result, str):
            try:
                tool_data = json.loads(result)
                for source in tool_data.get("sources", []):
                    if source not in sources:
                        sources.append(source)
            except json.JSONDecodeError:
                pass

        tool_result = {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": str(result),
        }
        messages.append({"role": "user", "content": [tool_result]})


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

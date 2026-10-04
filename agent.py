import os

from anthropic import Anthropic
from dotenv import load_dotenv
from rag import retrieve_documents, collection

load_dotenv()

client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))


def add_numbers(a, b):
    return a + b


def multiply_numbers(a, b):
    return a * b


def subtract_numbers(a, b):
    return a - b


def divide_numbers(a, b):
    return a / b


def search_knowledge(question):
    result = retrieve_documents(collection, question)
    if result:
        text = result[0]["text"]
        source = result[0]["metadata"]["document"]
        return f"Information: {text}\nSource: {source}"
    return "No relevant information found."


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


def main():
    question = input("You: ")

    messages = [{"role": "user", "content": question}]
    while True:
        response = client.messages.create(
            model="claude-sonnet-4-5", max_tokens=500, tools=tools, messages=messages
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_use = None
        for block in response.content:
            if block.type == "tool_use":
                tool_use = block
                break

        if tool_use is None:
            print("Claude", response.content[0].text)
            break
        print("Selected tool:", tool_use.name)
        print("Input:", tool_use.input)
        result = execute_tool(tool_use)
        print("Tool result:", result)
        tool_result = {
            "type": "tool_result",
            "tool_use_id": tool_use.id,
            "content": str(result),
        }
        messages.append({"role": "user", "content": [tool_result]})


if __name__ == "__main__":
    main()

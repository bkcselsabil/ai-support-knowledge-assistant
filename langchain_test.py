import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain.agents import create_agent
from rag import retrieve_documents, collection

load_dotenv()
llm = ChatAnthropic(model="claude-sonnet-4-5", api_key=os.getenv("ANTHROPIC_API_KEY"))


@tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@tool
def search_knowledge(question: str) -> str:
    """Search the company knowledge base for relevant information."""

    result = retrieve_documents(collection, question, threshold=0.8)
    # print("RAG RESULT:", result)
    if result:
        text = result[0]["text"]
        source = result[0]["metadata"]["document"]

        return f"Information : {text}\nSource: {source}"
    return "No relevant information found."


agent = create_agent(
    model=llm,
    tools=[add_numbers, multiply_numbers, search_knowledge],
    system_prompt="You are a helpful calculator assistant",
)


llm_with_tools = llm.bind_tools([add_numbers])

system_message = SystemMessage(
    content="You are an AI engineering teacher. Explain concepts step by step"
)


def interactive_chat():
    messages = []

    while True:
        question = input("You= ")

        if question.lower() == "quit":
            break

        human_message = HumanMessage(content=question)
        messages.append(human_message)

        response = llm_with_tools.invoke([system_message] + messages)
        messages.append(response)


def run_agent():
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "What is the vacation policy?"}]}
    )
    print("result", result["messages"][-1].content)


if __name__ == "__main__":
    interactive_chat()
    run_agent()

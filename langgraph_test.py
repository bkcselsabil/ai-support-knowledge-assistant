import os
from typing import Literal, NotRequired,TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from rag import collection, retrieve_documents

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-5", api_key=os.getenv("ANTHROPIC_API_KEY"))


class RouteDecision(BaseModel):
    route: Literal["rag", "calculator", "rag_then_calculate"]


class Calculation(BaseModel):
    number1: float
    operation: Literal["add", "subtract", "multiply", "divide"]
    number2: float


class State(TypedDict):
    message: str
    result: str
    route: str
    context: NotRequired[str]

def llm_router(state: State):
    response = llm.with_structured_output(RouteDecision).invoke(f"""
        Classify the user's message into exactly one route:

        - rag: questions that can be answered using company information
        or policies, without needing an additional calculation.

        - calculator: arithmetic questions that can be answered using
        the numbers and operation in the question itself.

        - rag_then_calculate: questions that require company information
        from documents first, followed by a calculation.

        Examples:
        - "How many vacation days do employees receive?" -> rag
        - "What is 50 minus 18?" -> calculator
        - "An employee used 12 vacation days. How many remain?"
        -> rag_then_calculate

        User message:
        {state["message"]}
        """)

    return {"route": response.route}


def rag_node(state: State):
    question = state["message"]

    retrieved = retrieve_context_node(state)
    context = retrieved["context"]

    if not context:
        return {
            "result": "I couldn't find relevant information in the company documents."
        }

    response = llm.invoke(f"""Answer the question using only the context below.
        If the answer is not in the context, say you don't know.

        Context: {context}
        Question: {question}
        """)

    return {"result": response.content}


def retrieve_context_node(state: State):
    question = state["message"]

    # First attempt: use the default threshold of 0.7
    documents = retrieve_documents(collection, question)

    # Retry with 0.8 if the first attempt finds nothing
    if not documents:
        documents = retrieve_documents(
            collection,
            question,
            threshold=0.8,
        )

    # Return empty context if both attempts fail
    if not documents:
        return {"context": ""}

    context = "\n\n".join(doc["text"] for doc in documents)

    return {"context": context}


def missing_context_node(state: State):
    return {
        "result": "I couldn't find relevant information in the company documents."
    }

def calculate(number1: float, operation: str, number2: float):
    if operation == "add":
        return number1 + number2
    elif operation == "subtract":
        return number1 - number2
    elif operation == "multiply":
        return number1 * number2
    elif operation == "divide":
        if number2 == 0:
            return "Error: Cannot divide by zero"
        return number1 / number2
    else:
        return "Error: Unsupported operation."

def calculator_node(state: State):
    question = state["message"]
    context = state.get("context", "")

    calculation = llm.with_structured_output(Calculation).invoke(f"""
        Extract the mathematical operation and two numbers
        needed to answer the question.

        If the question requires a fact from the provided context,
        use that fact when extracting the numbers.

        If the context is empty, use the numbers in the question.

        Question: {question}

        Context: {context}
        """)

    result = calculate(
        calculation.number1,
        calculation.operation,
        calculation.number2,
    )
    return {"result": str(result)}

graph = StateGraph(State)

graph.add_node("router", llm_router)
graph.add_node("retrieve_context", retrieve_context_node)
graph.add_node("rag", rag_node)
graph.add_node("calculator", calculator_node)
graph.add_node("missing_context", missing_context_node)
graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router",
    lambda state: state["route"],
    {
        "rag": "rag",
        "calculator": "calculator",
        "rag_then_calculate": "retrieve_context",
    },
)
graph.add_conditional_edges(
    "retrieve_context",
    lambda state: (
        "calculator" if state.get("context") else "missing_context"
    ),
    {
        "calculator": "calculator",
        "missing_context": "missing_context",
    },
)
graph.add_edge("rag", END)
graph.add_edge("calculator", END)
graph.add_edge("missing_context", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"message": "What is 50 minus 18?", "result": "", "route": ""})
    print(result)

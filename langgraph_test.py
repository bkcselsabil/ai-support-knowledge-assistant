import os
from typing import Literal, TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel

from rag import collection, retrieve_documents

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-5", api_key=os.getenv("ANTHROPIC_API_KEY"))


class RouteDecision(BaseModel):
    route: Literal["rag", "calculator"]


class Calculation(BaseModel):
    number1: float
    operation: Literal["add", "subtract", "multiply", "divide"]
    number2: float


class State(TypedDict):
    message: str
    result: str
    route: str


def llm_router(state: State):
    response = llm.with_structured_output(RouteDecision).invoke(f"""
        Classify the user's message into exactly one route:

        - rag: questions about company information, policies, employees,
        vacation, remote work, or related topics.
        - calculator: questions requiring arithmetic calculations.

        User message:
        {state["message"]}
        """)

    return {"route": response.route}


def rag_node(state: State):
    question = state["message"]

    # First attempt: use the default threshold of 0.7
    documents = retrieve_documents(collection, question)

    # Second attempt: if nothing was found, try threshold 0.8
    if not documents:
        documents = retrieve_documents(
            collection,
            question,
            threshold=0.8,
        )

    # If both attempts fail, return the fallback message
    if not documents:
        return {
            "result": "I couldn't find relevant information in the company documents."
        }

    context = "\n\n".join(doc["text"] for doc in documents)

    response = llm.invoke(f"""Answer the question using only the context below.
        If the answer is not in the context, say you don't know.

        Context: {context}
        Question: {question}
        """)

    return {"result": response.content}


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
    calculation = llm.with_structured_output(Calculation).invoke(f"""
        Extract the mathematical operation and two numbers
        from this question.

        Question: {state["message"]}



        """)
    result = calculate(calculation.number1, calculation.operation, calculation.number2)
    return {"result": str(result)}


graph = StateGraph(State)


graph.add_node("router", llm_router)
graph.add_node("rag", rag_node)
graph.add_node("calculator", calculator_node)


graph.add_edge(START, "router")

graph.add_conditional_edges(
    "router", lambda state: state["route"], {"rag": "rag", "calculator": "calculator"}
)
graph.add_edge("rag", END)

graph.add_edge("calculator", END)
app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"message": "What is 50 minus 18?", "result": "", "route": ""})
    print(result)

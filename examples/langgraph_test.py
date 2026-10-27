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


class ContextAssessment(BaseModel):
    sufficient: bool
    reason: str


class State(TypedDict):
    message: str
    result: str
    route: str
    context: NotRequired[str]
    context_sufficient: NotRequired[bool]


def llm_router(state: State):
    response = llm.with_structured_output(RouteDecision).invoke(f"""
        Classify the user's message into exactly one route.

        - rag: Questions about company information or policies that
        can be answered from company documents without calculation.
        This includes questions about general vacation policy,
        employee benefits, and whether a policy detail is documented.
        Personal questions also belong here when they ask what the
        company documents say about an employee's entitlement.

        - calculator: Arithmetic questions that can be answered using
        only numbers and operations explicitly provided in the
        user's message. No company information is needed.

        - rag_then_calculate: Questions that require retrieving company
        information from documents before performing a calculation.

        Examples:
        - "How many vacation days do employees receive?" -> rag
        - "What is the company's vacation policy?" -> rag
        - "How much is my annual vacation allowance?" -> rag
        - "How many vacation days can I carry over?" -> rag
        - "What is 50 minus 18?" -> calculator
        - "I have 25 vacation days and used 8. How many remain?"
        -> calculator
        - "How many vacation days remain if the policy gives me
        30 days and I have used 8?" -> rag_then_calculate

        Important:
        Choose a route based on the information the question requires,
        not on whether the answer is available. The RAG process must
        determine whether the documents actually support the answer.
        Never assume that a general policy establishes a specific
        employee's personal entitlement.

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
    response = llm.invoke(f"""
    You are a company support assistant. Answer the user's question
    using only the information explicitly supported by the context.

    Rules:
    1. Do not invent policies, facts, or personal employee information.
    2. Distinguish general company policy from an individual employee's
    personal entitlement or circumstances.
    3. If the context describes a general policy but does not confirm
    the user's personal situation, explain what the policy says and
    clearly state that you cannot confirm the user's individual
    entitlement from the available documents.
    4. If the requested information is not present in the context,
    say that the knowledge base does not contain the information.
    5. Do not present assumptions as facts.
    6. Keep your answer clear and concise.

    Context:
    {context}

    Question:
    {question}
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


def assess_context_node(state: State):
    question = state["message"]
    context = state.get("context", "")

    assessment = llm.with_structured_output(ContextAssessment).invoke(
        f"""Determine whether the context contains all facts needed
        to answer the question accurately.

        Do not assume missing facts or use outside knowledge.
        If a required fact is missing, mark the context insufficient.

        Question: {question}

        Context: {context}
        """
    )

    return {"context_sufficient": assessment.sufficient}

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
graph.add_node("assess_context", assess_context_node)
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
        "assess_context" if state.get("context") else "missing_context"
    ),
    {
        "assess_context": "assess_context",
        "missing_context": "missing_context",
    },
)
graph.add_conditional_edges(
    "assess_context",
    lambda state: (
        "calculator" if state.get("context_sufficient") else "missing_context"
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

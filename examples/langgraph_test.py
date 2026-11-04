import os
import json
from typing import Literal, NotRequired,TypedDict

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, START, END
from pydantic import BaseModel
from agent import list_knowledge_documents, get_document_overview
from rag import collection, retrieve_documents

load_dotenv()

llm = ChatAnthropic(model="claude-sonnet-4-5", api_key=os.getenv("ANTHROPIC_API_KEY"))


class RouteDecision(BaseModel):
    route: Literal[
        "rag",
        "calculator",
        "rag_then_calculate",
        "list_documents",
        "document_overview",
    ]

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
    sources: NotRequired[list[dict]]
    history: NotRequired[list[dict]]
    search_question: NotRequired[str]
    tenant_id: NotRequired[str]
    document_name: NotRequired[str | None]
    document_not_found: NotRequired[bool]

class RewrittenQuestion(BaseModel):
    question: str

class DocumentSelection(BaseModel):
    specific_document_requested: bool
    document_name: str | None = None

def llm_router(state: State):
    history = state.get("history", [])

    recent_history = history[-6:]

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
        if isinstance(message.get("content"), str)
    )
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
        - list_documents: Questions asking which documents are
        available in the company knowledge base, or requesting
        a list of available documents.
        - document_overview: Requests to summarize, explain, or provide a complete overview of an entire specific document.
        Examples:
        - "What documents are available?" -> list_documents
        - "List all company documents." -> list_documents
        - "What documents do you have?" -> list_documents
        Do not use rag for questions asking only for the
        document inventory.
        - "How many vacation days do employees receive?" -> rag
        - "What is the company's vacation policy?" -> rag
        - "How much is my annual vacation allowance?" -> rag
        - "How many vacation days can I carry over?" -> rag
        - "What is 50 minus 18?" -> calculator
        - "I have 25 vacation days and used 8. How many remain?"
        -> calculator
        - "How many vacation days remain if the policy gives me
        30 days and I have used 8?" -> rag_then_calculate
        - "Summarize the Remote Work Policy." -> document_overview
        - "Give me a complete overview of the Vacation Policy."
            -> document_overview
        - "What topics are covered in the employee handbook?"
            -> document_overview

        Do not use document_overview for a specific factual
        question about a document. Use rag instead.
        Important:
        Choose a route based on the information the question requires,
        not on whether the answer is available. The RAG process must
        determine whether the documents actually support the answer.
        Never assume that a general policy establishes a specific
        employee's personal entitlement.

        Previous conversation:
        {history_text}

        Current user message:
        {state["message"]}

        Use the previous conversation to understand references
        such as "that", "it", or "those days".

        Classify the current request based on the information
        needed to answer it.

        Do not treat previous assistant answers as verified
        company policy. Company facts must still be supported
        by retrieved documents.
    """)

    return {"route": response.route}

def rag_node(state: State):
    question = state["message"]
    context = state.get("context", "")
    history = state.get("history", [])

    recent_history = history[-6:]

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
        if isinstance(message.get("content"), str)
    )
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
    7. If the user asks to simplify or rephrase information,
    provide the simpler explanation directly.

    8. Do not ask for clarification when the retrieved context
    already contains enough information to answer.

    9. For follow-up questions, use the conversation history
    to understand what the user is referring to.

    10. Never introduce company facts that are not supported
        by the retrieved context.

    Previous conversation:
    {history_text}

    Retrieved company information:
    {context}

    Current user question:
    {question}

    Answer the current question directly, using the previous
    conversation to understand references and the retrieved
    company information to support factual claims.
        """)

    return {"result": response.content}

def rewrite_question_node(state: State):
    history = state.get("history", [])
    question = state["message"]

    recent_history = history[-6:]

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in recent_history
        if isinstance(message.get("content"), str)
    )

    if not history_text:
        return {"search_question": question}

    response = llm.with_structured_output(
        RewrittenQuestion
    ).invoke(f"""
        Rewrite the current user question into a standalone
        question using the previous conversation.

        Rules:
        - Preserve the user's original intent.
        - Resolve references such as "that", "it", or "those".
        - Do not invent company policies or facts.
        - Do not answer the question.
        - If the question is already standalone, keep it unchanged.

        Previous conversation:
        {history_text}

        Current question:
        {question}
    """)

    return {"search_question": response.question}

def retrieve_context_node(state: State):
    question = state.get("search_question", state["message"])

    # First attempt: use the default threshold of 0.9
    documents = retrieve_documents(
    collection,
    question,
    document_name=state.get("document_name"),
    tenant_id=state.get("tenant_id", "local-dev"),)

    # Retry with a more permissive threshold of 1.1

    if not documents:
        documents = retrieve_documents(
            collection,
            question,
            threshold=1.1,
        )

    # Return empty context if both attempts fail
    if not documents:
        return {
        "context": "",
        "sources": [],
        }

    context = "\n\n".join(doc["text"] for doc in documents)

    sources = []

    for doc in documents:
        meta = doc["metadata"]

        source = {
            "title": meta.get("title") or meta.get("document", "Unknown"),
            "document": meta.get("document", "Unknown"),
            "url": meta.get("url"),
            "source": meta.get("source", "local"),
        }

        if source not in sources:
            sources.append(source)

    return {
        "context": context,
        "sources": sources,
    }


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
    if state["route"] == "rag_then_calculate":
        return {
            "result": (
                f"Based on the retrieved company policy, "
                f"the calculated result is {result}. "
                "This calculation uses the general policy "
                "and does not confirm your personal entitlement."
            )
        }

    return {"result": str(result)}

def list_documents_node(state: State):
    tenant_id = state.get("tenant_id", "local-dev")

    inventory_json = list_knowledge_documents(
        tenant_id=tenant_id
    )

    inventory = json.loads(inventory_json)

    documents = inventory["documents"]

    if not documents:
        return {
            "result": "No documents are available in the knowledge base.",
            "sources": [],
        }

    lines = []

    for doc in documents:
        title = doc.get("title") or doc["document"]
        category = doc.get("category", "Unknown")

        lines.append(f"- {title} (Category: {category})")

    return {
        "result": (
            f"Available documents ({len(documents)}):\n"
            + "\n".join(lines)
        ),
        "sources": [],
    }

def select_document_node(state: State):
    from agent import list_knowledge_documents

    inventory = json.loads(
        list_knowledge_documents(
            tenant_id=state.get("tenant_id", "local-dev")
        )
    )

    documents = inventory["documents"]

    selection = llm.with_structured_output(
        DocumentSelection
    ).invoke(f"""
        Identify whether the user explicitly requests
        information from a specific document.

        Available documents:
        {json.dumps(documents)}

        User question:
        {state.get("search_question", state["message"])}

        Rules:
        - Set specific_document_requested to true when the user
        explicitly names or refers to a particular document.
        - Match document titles or filenames against the inventory.
        - Return the exact 'document' identifier when a match exists.
        - Never invent document identifiers.
        - If no particular document is requested, return false
        and document_name null.
        - If a particular document is requested but does not exist,
        return true and document_name null.
    """)

    valid_names = {doc["document"] for doc in documents}

    document_name = selection.document_name

    if document_name not in valid_names:
        document_name = None

    return {
        "document_name": document_name,
        "document_not_found": (
            selection.specific_document_requested
            and document_name is None
        ),
    }

def missing_document_node(state: State):
    return {
        "result": (
            "The requested document was not found "
            "in the company knowledge base."
        ),
        "sources": [],
    }

def document_overview_node(state: State):
    document_name = state.get("document_name")

    if not document_name:
        return {
            "result": "Please specify which document you want summarized.",
            "sources": [],
        }

    overview_json = get_document_overview(
        document_name=document_name,
        tenant_id=state.get("tenant_id", "local-dev"),
    )

    overview = json.loads(overview_json)

    return {
        "result": overview["information"],
        "sources": overview.get("sources", []),
    }
graph = StateGraph(State)

graph.add_node("router", llm_router)
graph.add_node("document_overview", document_overview_node)
graph.add_node("missing_document", missing_document_node)
graph.add_node("list_documents", list_documents_node)
graph.add_node("select_document", select_document_node)
graph.add_node("rewrite_question", rewrite_question_node)
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
        "rag": "rewrite_question",
        "calculator": "calculator",
        "rag_then_calculate": "rewrite_question",
        "list_documents": "list_documents",
        "document_overview": "rewrite_question",
    },
)
graph.add_conditional_edges(
    "retrieve_context",
    lambda state: (
        "missing_context"
        if not state.get("context")
        else "rag"
        if state["route"] == "rag"
        else "assess_context"
    ),
    {
        "rag": "rag",
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
graph.add_edge("missing_document", END)
graph.add_edge("calculator", END)
graph.add_edge("missing_context", END)
graph.add_edge("rewrite_question", "select_document")
graph.add_conditional_edges(
    "select_document",
    lambda state: (
        "missing_document"
        if state.get("document_not_found")
        else "document_overview"
        if state["route"] == "document_overview"
        else "retrieve_context"
    ),
    {
        "missing_document": "missing_document",
        "document_overview": "document_overview",
        "retrieve_context": "retrieve_context",
    },
)
graph.add_edge("list_documents", END)
graph.add_edge("document_overview", END)

app = graph.compile()
if __name__ == "__main__":
    result = app.invoke({
        "message": "Give me a complete overview of a document",
        "result": "",
        "route": "",
        "tenant_id": "local-dev",
    })

    print("Route:", result["route"])
    print("Selected document:", result.get("document_name"))
    print("Answer:", result["result"])
    print("Sources:", result.get("sources", []))
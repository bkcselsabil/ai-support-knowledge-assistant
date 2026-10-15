import pytest
from langgraph_test import app, llm_router
from types import SimpleNamespace


def test_calculator_question():
    result = app.invoke({
        "message": "What is 50 minus 18?",
        "result": "",
        "route": "",
    })

    assert result["route"] == "calculator"
    assert result["result"] == "32.0"

def test_rag_question():
    result = app.invoke({
        "message": "How many vacation days do employees get?",
        "result": "",
        "route" : "",
    })
    assert result["route"] == "rag"
    assert "30" in result["result"]



@pytest.mark.parametrize("expected_route", ["calculator", "rag"])

def test_router_returns_expected_route(mocker, expected_route):
    mock_llm = mocker.patch("langgraph_test.llm")

    mock_llm.with_structured_output.return_value.invoke.return_value.route = expected_route

    result = llm_router({
        "message": "Test question",
        "result": "",
        "route": "",
    })

    assert result["route"] == expected_route

def test_graph_routes_to_calculator(mocker):
    mock_llm = mocker.patch("langgraph_test.llm")

    router_response = SimpleNamespace(route="calculator")

    calculation_response = SimpleNamespace(
        number1=50,
        operation="subtract",
        number2=18,
    )

    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        router_response,
        calculation_response,
    ]

    result = app.invoke({
        "message": "What is 50 minus 18?",
        "result": "",
        "route": "",
    })

    assert result["route"] == "calculator"
    assert result["result"] == "32"



def test_graph_routes_to_rag(mocker):
    mock_llm = mocker.patch("langgraph_test.llm")
    mock_retrieve = mocker.patch("langgraph_test.retrieve_documents")

    # Control the documents returned by retrieval
    mock_retrieve.return_value = [
        {"text": "Employees receive 30 vacation days per year."}
    ]

    # Simulate Claude choosing the RAG route
    mock_llm.with_structured_output.return_value.invoke.return_value.route = "rag"

    # Simulate Claude's answer using the retrieved context
    mock_llm.invoke.return_value.content = (
        "Employees receive 30 vacation days per year."
    )

    result = app.invoke({
        "message": "How many vacation days do employees get?",
        "result": "",
        "route": "",
    })

    assert result["route"] == "rag"
    assert "30" in result["result"]
    mock_retrieve.assert_called_once()



def test_rag_returns_fallback_when_no_documents(mocker):
    mock_llm = mocker.patch("langgraph_test.llm")
    mock_retrieve = mocker.patch("langgraph_test.retrieve_documents")

    mock_retrieve.return_value = []
    mock_llm.with_structured_output.return_value.invoke.return_value.route = "rag"

    result = app.invoke({
        "message": "What is the company's office in Antarctica?",
        "result": "",
        "route": "",
    })

    assert result["route"] == "rag"
    assert "couldn't find relevant information" in result["result"]
    assert mock_retrieve.call_count == 2
    mock_llm.invoke.assert_not_called()



def test_rag_retries_with_higher_threshold(mocker):
    mock_llm = mocker.patch("langgraph_test.llm")
    mock_retrieve = mocker.patch("langgraph_test.retrieve_documents")

    question = (
        "An employee has taken 12 vacation days. "
        "How many vacation days do they have left?"
    )

    # First attempt finds nothing; the retry finds the vacation document.
    mock_retrieve.side_effect = [
        [],
        [
            {
                "text": "Employees receive 30 days of annual vacation.",
                "distance": 0.7625,
                "metadata": {"document": "employee_handbook.txt"},
            }
        ],
    ]

    # Simulate the router choosing RAG.
    mock_llm.with_structured_output.return_value.invoke.return_value.route = "rag"

    # Simulate the answer generated from the retrieved document.
    mock_llm.invoke.return_value.content = (
        "The employee has 18 vacation days remaining."
    )

    result = app.invoke({
        "message": question,
        "result": "",
        "route": "",
    })

    assert result["route"] == "rag"
    assert "18" in result["result"]
    assert mock_retrieve.call_count == 2
    assert mock_retrieve.call_args_list[1].kwargs["threshold"] == 0.8


def test_graph_retrieves_context_then_calculates(mocker):
    mock_llm = mocker.patch("langgraph_test.llm")
    mock_retrieve = mocker.patch("langgraph_test.retrieve_documents")

    # Simulate the router selecting the multi-step route,
    # followed by Claude extracting 30 - 12.
    router_response = SimpleNamespace(route="rag_then_calculate")
    calculation_response = SimpleNamespace(
        number1=30,
        operation="subtract",
        number2=12,
    )

    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        router_response,
        calculation_response,
    ]

    # First retrieval fails; the retry finds the vacation policy.
    mock_retrieve.side_effect = [
        [],
        [
            {
                "text": "Employees receive 30 days of annual vacation.",
                "distance": 0.7625,
                "metadata": {"document": "employee_handbook.txt"},
            }
        ],
    ]

    result = app.invoke({
        "message": (
            "An employee has taken 12 vacation days. "
            "How many vacation days do they have left?"
        ),
        "result": "",
        "route": "",
    })

    assert result["route"] == "rag_then_calculate"
    assert "30 days of annual vacation" in result["context"]
    assert result["result"] == "18"
    assert mock_retrieve.call_count == 2
    assert mock_retrieve.call_args_list[1].kwargs["threshold"] == 0.8


def test_multi_step_returns_fallback_when_no_documents(mocker):
    mock_llm = mocker.patch("langgraph_test.llm")
    mock_retrieve = mocker.patch("langgraph_test.retrieve_documents")

    router_response = SimpleNamespace(route="rag_then_calculate")
    calculation_response = SimpleNamespace(
        number1=30,
        operation="subtract",
        number2=12,
    )

    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        router_response,
        calculation_response,
    ]

    # Neither retrieval attempt finds a document.
    mock_retrieve.side_effect = [[], []]

    result = app.invoke({
        "message": (
            "An employee has taken 12 vacation days. "
            "How many vacation days do they have left?"
        ),
        "result": "",
        "route": "",
    })

    assert result["route"] == "rag_then_calculate"
    assert result["result"] == (
        "I couldn't find relevant information in the company documents."
    )
    assert mock_retrieve.call_count == 2
    assert mock_llm.with_structured_output.return_value.invoke.call_count == 1

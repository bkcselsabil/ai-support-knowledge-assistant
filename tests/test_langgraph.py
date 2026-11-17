import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from examples.langgraph_test import app, llm_router, assess_context_node
from types import SimpleNamespace

def test_calculator_question(mocker):
    from examples.langgraph_test import RouteDecision, Calculation

    mock_llm = mocker.Mock()

    def mock_structured_output(schema):
        mock_chain = mocker.Mock()

        if schema is RouteDecision:
            mock_chain.invoke.return_value = RouteDecision(
                route="calculator"
            )

        elif schema is Calculation:
            mock_chain.invoke.return_value = Calculation(
                number1=50,
                operation="subtract",
                number2=18,
            )

        return mock_chain

    mock_llm.with_structured_output.side_effect = mock_structured_output

    mocker.patch(
        "examples.langgraph_test.llm",
        mock_llm,
    )

    result = app.invoke({
        "message": "What is 50 minus 18?",
        "result": "",
        "route": "",
        "tenant_id": "local-dev",
    })

    assert result["route"] == "calculator"
    assert result["result"] == "32.0"
    

def test_rag_question(mocker):
    from examples.langgraph_test import RouteDecision

    # Mock Claude
    mock_llm = mocker.Mock()

    from examples.langgraph_test import DocumentSelection

    def mock_structured_output(schema):
        mock_chain = mocker.Mock()

        if schema is RouteDecision:
            mock_chain.invoke.return_value = RouteDecision(
                route="rag"
            )

        elif schema is DocumentSelection:
            mock_chain.invoke.return_value = DocumentSelection(
                specific_document_requested=False,
                document_name=None,
            )

        else:
            raise AssertionError(f"Unexpected schema: {schema}")

        return mock_chain

    mock_llm.with_structured_output.side_effect = mock_structured_output

    mock_llm.invoke.return_value = SimpleNamespace(
        content="Full-time employees receive 30 vacation days."
    )

    mocker.patch("examples.langgraph_test.llm", mock_llm)

    # Mock ChromaDB retrieval
    mock_retrieve = mocker.patch(
        "examples.langgraph_test.retrieve_documents",
        return_value=[
            {
                "text": "Full-time employees receive 30 vacation days.",
                "metadata": {
                    "title": "Vacation Policy",
                    "document": "vacation_policy.txt",
                    "source": "local",
                },
            }
        ],
    )

    result = app.invoke({
        "message": "How many vacation days do employees get?",
        "result": "",
        "route": "",
        "tenant_id": "local-dev",
    })

    assert result["route"] == "rag"
    assert "30" in result["result"]
    assert result["sources"][0]["title"] == "Vacation Policy"
    mock_retrieve.assert_called_once()



@pytest.mark.parametrize("expected_route", ["calculator", "rag"])

def test_router_returns_expected_route(mocker, expected_route):
    mock_llm = mocker.patch("examples.langgraph_test.llm")

    mock_llm.with_structured_output.return_value.invoke.return_value.route = expected_route

    result = llm_router({
        "message": "Test question",
        "result": "",
        "route": "",
        "tenant_id": "local-dev",
    })

    assert result["route"] == expected_route

def test_graph_routes_to_calculator(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")

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
        "tenant_id": "local-dev",
    })

    assert result["route"] == "calculator"
    assert result["result"] == "32"



def test_graph_routes_to_rag(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")
    mock_retrieve = mocker.patch("examples.langgraph_test.retrieve_documents")

    # Control the documents returned by retrieval
    mock_retrieve.return_value = [
        {
            "text": "Employees receive 30 vacation days per year.",
            "distance": 0.4,
            "metadata": {
                "document": "employee_handbook.txt",
                "category": "HR",
                "source": "local",
            },
        }
    ]

    # Simulate Claude choosing the RAG route
    from types import SimpleNamespace

    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        SimpleNamespace(route="rag"),
        SimpleNamespace(
            specific_document_requested=False,
            document_name=None,
        ),
    ]
    # Simulate Claude's answer using the retrieved context
    mock_llm.invoke.return_value.content = (
        "Employees receive 30 vacation days per year."
    )

    result = app.invoke({
        "message": "How many vacation days do employees get?",
        "result": "",
        "route": "",
        "tenant_id": "local-dev",
    })

    assert result["route"] == "rag"
    assert "30" in result["result"]
    mock_retrieve.assert_called_once()



def test_rag_returns_fallback_when_no_documents(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")
    mock_retrieve = mocker.patch("examples.langgraph_test.retrieve_documents")

    mock_retrieve.return_value = []
    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        SimpleNamespace(route="rag"),
        SimpleNamespace(
            specific_document_requested=False,
            document_name=None,
        ),
    ]
    result = app.invoke({
        "message": "What is the company's office in Antarctica?",
        "result": "",
        "route": "",
        "tenant_id": "local-dev",
    })

    assert result["route"] == "rag"
    assert "couldn't find relevant information" in result["result"]
    assert mock_retrieve.call_count == 2
    mock_llm.invoke.assert_not_called()



def test_rag_retries_with_higher_threshold(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")
    mock_retrieve = mocker.patch("examples.langgraph_test.retrieve_documents")

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
    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        SimpleNamespace(route="rag"),
        SimpleNamespace(
            specific_document_requested=False,
            document_name=None,
        ),
    ]
    # Simulate the answer generated from the retrieved document.
    mock_llm.invoke.return_value.content = (
        "The employee has 18 vacation days remaining."
    )

    result = app.invoke({
        "message": question,
        "result": "",
        "route": "",
        "tenant_id": "local-dev",
    })

    assert result["route"] == "rag"
    assert "18" in result["result"]
    assert mock_retrieve.call_count == 2
    assert mock_retrieve.call_args_list[1].kwargs["threshold"] == 1.1


def test_graph_retrieves_context_then_calculates(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")
    mock_retrieve = mocker.patch("examples.langgraph_test.retrieve_documents")

    # Simulate the router selecting the multi-step route,
    # followed by Claude extracting 30 - 12.
    router_response = SimpleNamespace(route="rag_then_calculate")

    assessment_response = SimpleNamespace(
    sufficient=True,
    reason="The context contains the annual vacation allowance.",
    )

    calculation_response = SimpleNamespace(
    number1=30,
    operation="subtract",
    number2=12,
    )

    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        router_response,
        SimpleNamespace(
            specific_document_requested=False,
            document_name=None,
        ),
        assessment_response,
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
        "tenant_id": "local-dev",
    })

    assert result["route"] == "rag_then_calculate"
    assert "30 days of annual vacation" in result["context"]
    assert "calculated result is 18" in result["result"]
    assert "does not confirm your personal entitlement" in result["result"]
    assert mock_retrieve.call_count == 2
    assert mock_retrieve.call_args_list[1].kwargs["threshold"] == 1.1


def test_multi_step_returns_fallback_when_no_documents(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")
    mock_retrieve = mocker.patch("examples.langgraph_test.retrieve_documents")

    router_response = SimpleNamespace(route="rag_then_calculate")
    calculation_response = SimpleNamespace(
        number1=30,
        operation="subtract",
        number2=12,
    )

    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        router_response,
        SimpleNamespace(
            specific_document_requested=False,
            document_name=None,
        ),
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
        "tenant_id": "local-dev",
    })

    assert result["route"] == "rag_then_calculate"
    assert result["result"] == (
        "I couldn't find relevant information in the company documents."
    )
    assert mock_retrieve.call_count == 2
    assert mock_llm.with_structured_output.return_value.invoke.call_count == 2

def test_multi_step_does_not_calculate_when_context_lacks_required_fact(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")
    mock_retrieve = mocker.patch("examples.langgraph_test.retrieve_documents")

    router_response = SimpleNamespace(route="rag_then_calculate")

    assessment_response = SimpleNamespace(
    sufficient=False,
    reason="The context does not contain the annual vacation allowance.",
)

    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        router_response,
        SimpleNamespace(
            specific_document_requested=False,
            document_name=None,
        ),
        assessment_response,
    ]

    mock_retrieve.return_value = [
        {
            "text": "Employees submit vacation requests through the HR portal.",
            "distance": 0.5,
            "metadata": {"document": "employee_handbook.txt"},
        }
    ]

    result = app.invoke({
        "message": (
            "An employee has taken 12 vacation days. "
            "How many vacation days do they have left?"
        ),
        "result": "",
        "route": "",
        "tenant_id": "local-dev",
    })

    assert result["result"] == (
        "I couldn't find relevant information in the company documents."
    )
    assert mock_llm.with_structured_output.return_value.invoke.call_count == 3
    mock_llm.invoke.assert_not_called()

def test_assess_context_returns_true_when_context_is_sufficient(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")

    mock_llm.with_structured_output.return_value.invoke.return_value = (
        SimpleNamespace(
            sufficient=True,
            reason="The annual vacation allowance is stated.",
        )
    )

    result = assess_context_node({
        "message": "How many vacation days remain after taking 12?",
        "result": "",
        "route": "rag_then_calculate",
        "context": "Employees receive 30 days of annual vacation.",
    })

    assert result["context_sufficient"] is True

def test_assess_context_returns_false_when_context_is_insufficient(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")

    mock_llm.with_structured_output.return_value.invoke.return_value = (
        SimpleNamespace(
            sufficient=False,
            reason="The annual vacation allowance is missing.",
        )
    )

    result = assess_context_node({
        "message": "How many vacation days remain after taking 12?",
        "result": "",
        "route": "rag_then_calculate",
        "context": "Employees submit vacation requests through the HR portal.",
    })

    assert result["context_sufficient"] is False

def test_rag_does_not_assume_part_time_entitlement(mocker):
    mock_llm = mocker.patch("examples.langgraph_test.llm")
    mock_retrieve = mocker.patch(
        "examples.langgraph_test.retrieve_documents"
    )

    mock_retrieve.return_value = [
        {
            "text": "Full-time employees receive 30 days of annual vacation.",
            "distance": 0.4,
            "metadata": {"document": "employee_handbook.txt"},
        }
    ]

    mock_llm.with_structured_output.return_value.invoke.side_effect = [
        SimpleNamespace(route="rag"),
        SimpleNamespace(
            specific_document_requested=False,
            document_name=None,
        ),
    ]
    mock_llm.invoke.return_value.content = (
        "The handbook specifies 30 days for full-time employees, "
        "but it does not specify the allowance for part-time employees. "
        "Please contact HR to confirm the applicable entitlement."
    )

    result = app.invoke({
        "message": (
            "I am a part-time employee. How many vacation days "
            "am I entitled to according to the company handbook?"
        ),
        "result": "",
        "route": "",
        "tenant_id": "local-dev",
    })

    answer = result["result"].lower()

    assert "30 days" in answer
    assert "part-time" in answer
    assert "hr" in answer
    mock_retrieve.assert_called_once()

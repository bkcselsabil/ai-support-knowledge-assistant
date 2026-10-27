import json
from types import SimpleNamespace

import pytest

import agent


@pytest.mark.parametrize(
    "function,a,b,expected",
    [
        (agent.add_numbers, 2, 3, 5),
        (agent.multiply_numbers, 2, 3, 6),
        (agent.subtract_numbers, 5, 3, 2),
        (agent.divide_numbers, 6, 2, 3),
        (agent.divide_numbers, 5, 2, 2.5),
    ],
)
def test_math_tools(function, a, b, expected):
    assert function(a, b) == expected


def test_divide_by_zero():
    assert agent.divide_numbers(10, 0) == "Cannot divide by zero."


def test_execute_unknown_tool():
    tool_use = SimpleNamespace(
        name="unknown_tool",
        input={},
    )

    assert agent.execute_tool(tool_use) == "Unknown tool"


def test_execute_tool_returns_result():
    tool_use = SimpleNamespace(
        name="add_numbers",
        input={"a": 4, "b": 6},
    )

    assert agent.execute_tool(tool_use) == 10


def test_execute_tool_handles_errors():
    tool_use = SimpleNamespace(
        name="add_numbers",
        input={"a": 4},
    )

    result = agent.execute_tool(tool_use)

    assert result.startswith("Tool error:")


def test_search_knowledge_returns_documents(mocker):
    mocker.patch(
        "agent.retrieve_documents",
        return_value=[
            {
                "text": "Employees receive 30 days of annual vacation.",
                "metadata": {
                    "document": "employee_handbook.txt",
                    "category": "HR",
                    "source": "local",
                },
            }
        ],
    )

    result = json.loads(agent.search_knowledge("How many vacation days?"))

    assert "30 days" in result["information"]
    assert result["sources"][0]["document"] == "employee_handbook.txt"


def test_search_knowledge_returns_empty_result(mocker):
    mocker.patch("agent.retrieve_documents", return_value=[])

    result = json.loads(agent.search_knowledge("Unknown policy"))

    assert result["information"] == "No relevant information found."
    assert result["sources"] == []
def test_run_agent_returns_final_answer(mocker):
    mock_create = mocker.patch("agent.client.messages.create")

    mock_create.return_value.content = [
        SimpleNamespace(
            type="text",
            text="Hello! How can I help you?",
        )
    ]

    result = agent.run_agent("Hello")

    assert result["answer"] == "Hello! How can I help you?"
    assert result["sources"] == []
    mock_create.assert_called_once()


def test_run_agent_executes_tool_and_returns_sources(mocker):
    mock_create = mocker.patch("agent.client.messages.create")

    tool_use = SimpleNamespace(
        type="tool_use",
        id="tool_123",
        name="search_knowledge",
        input={"question": "How many vacation days?"},
    )

    first_response = SimpleNamespace(content=[tool_use])

    second_response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text="Employees receive 30 vacation days.",
            )
        ]
    )

    mock_create.side_effect = [first_response, second_response]

    mocker.patch(
        "agent.execute_tool",
        return_value=json.dumps({
            "information": "Employees receive 30 vacation days.",
            "sources": [
                {
                    "title": "Employee Handbook",
                    "document": "employee_handbook.txt",
                    "url": None,
                    "source": "local",
                }
            ],
        }),
    )

    result = agent.run_agent("How many vacation days?")

    assert result["answer"] == "Employees receive 30 vacation days."
    assert len(result["sources"]) == 1
    assert result["sources"][0]["document"] == "employee_handbook.txt"
    assert mock_create.call_count == 2

def test_run_agent_executes_multiple_tool_calls(monkeypatch):
    first_tool = SimpleNamespace(
        type="tool_use",
        id="tool_1",
        name="add_numbers",
        input={"a": 2, "b": 3},
    )

    second_tool = SimpleNamespace(
        type="tool_use",
        id="tool_2",
        name="multiply_numbers",
        input={"a": 4, "b": 5},
    )

    tool_response = SimpleNamespace(
        content=[first_tool, second_tool]
    )

    final_response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text="The calculations are 5 and 20.",
            )
        ]
    )

    responses = iter([tool_response, final_response])

    monkeypatch.setattr(
        agent.client.messages,
        "create",
        lambda **kwargs: next(responses),
    )

    result = agent.run_agent("Calculate two values")

    assert result["answer"] == "The calculations are 5 and 20."
    assert len(result["messages"]) == 4

    tool_results = result["messages"][2]["content"]

    assert len(tool_results) == 2
    assert tool_results[0]["tool_use_id"] == "tool_1"
    assert tool_results[0]["content"] == "5"
    assert tool_results[1]["tool_use_id"] == "tool_2"
    assert tool_results[1]["content"] == "20"

def test_execute_tool_returns_unknown_tool_message():
    tool_use = SimpleNamespace(
        name="unknown_tool",
        input={},
    )

    result = agent.execute_tool(tool_use)

    assert result == "Unknown tool"

def test_execute_tool_handles_execution_error(monkeypatch):
    def broken_tool(**kwargs):
        raise ValueError("Something went wrong")

    monkeypatch.setitem(
        agent.tool_functions,
        "broken_tool",
        broken_tool,
    )

    tool_use = SimpleNamespace(
        name="broken_tool",
        input={},
    )

    result = agent.execute_tool(tool_use)

    assert result == "Tool error: Something went wrong"

def test_run_agent_recovers_after_tool_error(mocker):
    mock_create = mocker.patch("agent.client.messages.create")

    tool_use = SimpleNamespace(
        type="tool_use",
        id="tool_error_123",
        name="add_numbers",
        input={"a": 1, "b": 2},
    )

    first_response = SimpleNamespace(content=[tool_use])

    second_response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text="I couldn't complete the calculation because the tool failed.",
            )
        ]
    )

    mock_create.side_effect = [first_response, second_response]

    mocker.patch(
        "agent.execute_tool",
        return_value="Tool error: Something went wrong",
    )

    result = agent.run_agent("Calculate 1 + 2")

    assert result["answer"] == (
        "I couldn't complete the calculation because the tool failed."
    )
    assert mock_create.call_count == 2

    tool_results = result["messages"][2]["content"]
    assert len(tool_results) == 1
    assert tool_results[0]["content"] == "Tool error: Something went wrong"
    
def test_run_agent_handles_missing_knowledge(mocker):
    mock_create = mocker.patch("agent.client.messages.create")

    tool_use = SimpleNamespace(
        type="tool_use",
        id="tool_missing_123",
        name="search_knowledge",
        input={"question": "What is the company's relocation allowance?"},
    )

    first_response = SimpleNamespace(content=[tool_use])

    final_response = SimpleNamespace(
        content=[
            SimpleNamespace(
                type="text",
                text="I couldn't find information about the relocation allowance in the company knowledge base.",
            )
        ]
    )

    mock_create.side_effect = [first_response, final_response]

    mocker.patch(
        "agent.execute_tool",
        return_value=json.dumps({
            "information": "No relevant information found.",
            "sources": [],
        }),
    )

    result = agent.run_agent(
        "What is the company's relocation allowance?"
    )

    assert "couldn't find information" in result["answer"].lower()
    assert result["sources"] == []
    assert mock_create.call_count == 2

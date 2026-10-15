
from rag import retrieve_documents, generate_answer, detect_category, main
from types import SimpleNamespace


def test_retrieve_documents_filters_by_threshold(mocker):
    mock_collection = mocker.Mock()

    mock_collection.query.return_value = {
        "distances": [[0.4, 0.8, 1.2]],
        "documents": [[
            "Close match",
            "Document at threshold",
            "Irrelevant document",
        ]],
        "metadatas": [[
            {"document": "relevant.txt"},
            {"document": "boundary.txt"},
            {"document": "irrelevant.txt"},
        ]],
    }

    results = retrieve_documents(
        mock_collection,
        "test question",
        threshold=0.8,
    )

    assert len(results) == 1
    assert results[0]["text"] == "Close match"



def test_retrieve_documents_applies_metadata_filters(mocker):
    mock_collection = mocker.Mock()

    # Simulate ChromaDB returning one relevant document
    mock_collection.query.return_value = {
        "distances": [[0.4]],
        "documents": [[
            "Employees receive 30 days of annual vacation."
        ]],
        "metadatas": [[{
            "document": "employee_handbook.txt",
            "category": "HR",
        }]],
    }

    results = retrieve_documents(
        mock_collection,
        "How many vacation days do employees get?",
        document_name="employee_handbook.txt",
        category="HR",
    )

    # Verify that both filters were sent to ChromaDB
    mock_collection.query.assert_called_once_with(
        query_texts=["How many vacation days do employees get?"],
        n_results=4,
        where={
            "document": "employee_handbook.txt",
            "category": "HR",
        },
    )

    assert len(results) == 1
    assert results[0]["metadata"]["category"] == "HR"


def test_generate_answer_returns_answer_and_source(mocker):
    mock_anthropic = mocker.patch("rag.Anthropic")

    mock_anthropic.return_value.messages.create.return_value.content = [
        SimpleNamespace(
            text="Employees receive 30 days of annual vacation."
        )
    ]
    question = "How many vacation days do employees get?"
    relevant_results = [
        {
            "text": "Employees receive 30 days of annual vacation.",
            "metadata": {
                "document": "employee_handbook.txt"
            },
        }
    ]
    answer, source = generate_answer(question, relevant_results)

    assert answer == "Employees receive 30 days of annual vacation."
    assert source == "employee_handbook.txt"

    mock_anthropic.return_value.messages.create.assert_called_once()


def test_detect_category_returns_category(mocker):
    mock_anthropic = mocker.patch("rag.Anthropic")

    # Simulate Claude returning a category with extra whitespace
    mock_anthropic.return_value.messages.create.return_value.content = [
        mocker.Mock(text=" HR\n")
    ]

    result = detect_category(
        "How many vacation days do employees get?"
    )

    assert result == "HR"
    mock_anthropic.return_value.messages.create.assert_called_once()


def test_main_when_no_documents_found(mocker, capsys):
    mocker.patch("builtins.input", return_value="What is the pet policy?")
    mocker.patch("rag.detect_category", return_value="None")
    mock_retrieve = mocker.patch("rag.retrieve_documents", return_value=[])

    main()

    output = capsys.readouterr().out

    assert "Detected category None" in output
    assert "Retrieved 0 document(s)" in output
    assert "I don't have enough information to answer this question." in output
    mock_retrieve.assert_called_once()





def test_main_when_documents_found(mocker, capsys):
    question = "How many vacation days do employees get?"

    mocker.patch("builtins.input", return_value=question)
    mocker.patch("rag.detect_category", return_value="HR")

    mock_retrieve = mocker.patch(
        "rag.retrieve_documents",
        return_value=[
            {
                "text": "Employees receive 30 days of annual vacation.",
                "metadata": {"document": "employee_handbook.txt"},
            }
        ],
    )

    mock_generate = mocker.patch(
        "rag.generate_answer",
        return_value=(
            "Employees receive 30 days of annual vacation.",
            "employee_handbook.txt",
        ),
    )

    main()

    output = capsys.readouterr().out

    assert "Detected category HR" in output
    assert "Retrieved 1 document(s)" in output
    assert "30 days of annual vacation" in output
    assert "Source: employee_handbook.txt" in output

    mock_retrieve.assert_called_once()
    mock_generate.assert_called_once_with(
        question,
        mock_retrieve.return_value,
    )

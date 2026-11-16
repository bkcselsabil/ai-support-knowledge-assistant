
from rag import retrieve_documents, generate_answer, detect_category, main
from types import SimpleNamespace
import pytest


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
        tenant_id="local-dev",
    )

    assert len(results) == 2
    assert results[0]["text"] == "Close match"
    assert results[1]["text"] == "Document at threshold"



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
        tenant_id="local-dev",
    )

    # Verify that both filters were sent to ChromaDB

    mock_collection.query.assert_called_once_with(
        query_texts=["How many vacation days do employees get?"],
        n_results=4,
        where={
            "$and": [
                {"tenant_id": {"$eq": "local-dev"}},
                {"document": {"$eq": "employee_handbook.txt"}},
                {"category": {"$eq": "HR"}},
            ]
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



@pytest.mark.parametrize(
    "claude_response, expected_category",
    [
        ("Company Policy", "Company Policy"),
        ("None", "None"),
    ],
)
def test_detect_category_handles_other_categories(
    mocker, claude_response, expected_category
):
    mock_anthropic = mocker.patch("rag.Anthropic")

    mock_anthropic.return_value.messages.create.return_value.content = [
        mocker.Mock(text=claude_response)
    ]

    result = detect_category("Can employees work remotely?")

    assert result == expected_category

def test_retrieval_filters_by_tenant():
    from unittest.mock import MagicMock

    from rag import retrieve_documents

    collection = MagicMock()
    collection.query.return_value = {
        "documents": [[]],
        "distances": [[]],
        "metadatas": [[]],
    }

    results = retrieve_documents(
        collection,
        "What is the vacation policy?",
        tenant_id="organization-A",
    )

    assert results == []

    collection.query.assert_called_once_with(
        query_texts=["What is the vacation policy?"],
        n_results=4,
        where={
            "tenant_id": {
                "$eq": "organization-A"
            }
        },
    )
    
def test_chroma_cross_tenant_isolation(tmp_path):
    import chromadb

    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma_test")
    )

    collection = client.create_collection(
        name="tenant_isolation_test"
    )

    collection.add(
        ids=["chunk-a", "chunk-b"],
        documents=[
            "Organization A has 30 vacation days.",
            "Organization B has 45 vacation days.",
        ],
        metadatas=[
            {"tenant_id": "organization-A"},
            {"tenant_id": "organization-B"},
        ],
    )

    results = collection.get(
        where={"tenant_id": "organization-A"},
        include=["documents", "metadatas"],
    )

    assert len(results["ids"]) == 1
    assert results["documents"] == [
        "Organization A has 30 vacation days."
    ]
    assert results["metadatas"][0]["tenant_id"] == "organization-A"
    
def test_retrieve_documents_enforces_tenant_filter():
    from unittest.mock import Mock
    from rag import retrieve_documents

    mock_collection = Mock()

    mock_collection.query.return_value = {
        "distances": [[0.2]],
        "documents": [["Organization A vacation policy"]],
        "metadatas": [[{
            "tenant_id": "organization-A",
            "document": "handbook.txt",
        }]],
    }

    results = retrieve_documents(
        collection=mock_collection,
        question="How many vacation days?",
        tenant_id="organization-A",
    )

    mock_collection.query.assert_called_once_with(
        query_texts=["How many vacation days?"],
        n_results=4,
        where={
            "tenant_id": {"$eq": "organization-A"}
        },
    )

    assert len(results) == 1
    assert results[0]["metadata"]["tenant_id"] == "organization-A"
    
    
def test_retrieve_documents_real_tenant_isolation(tmp_path):
    import chromadb
    from rag import retrieve_documents

    client = chromadb.PersistentClient(
        path=str(tmp_path / "chroma_retrieval")
    )

    collection = client.create_collection(
        name="retrieval_isolation_test"
    )

    collection.add(
        ids=["doc-a", "doc-b"],
        documents=[
            "Organization A employees receive 30 vacation days.",
            "Organization B employees receive 45 vacation days.",
        ],
        metadatas=[
            {
                "tenant_id": "organization-A",
                "document": "handbook.txt",
                "category": "HR",
            },
            {
                "tenant_id": "organization-B",
                "document": "handbook.txt",
                "category": "HR",
            },
        ],
    )

    results = retrieve_documents(
        collection=collection,
        question="How many vacation days do employees receive?",
        tenant_id="organization-A",
        threshold=2.0,
    )

    assert len(results) == 1
    assert results[0]["metadata"]["tenant_id"] == "organization-A"
    assert "30 vacation days" in results[0]["text"]
    assert "45 vacation days" not in results[0]["text"]
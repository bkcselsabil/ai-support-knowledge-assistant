
from rag import retrieve_documents


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

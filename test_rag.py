
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

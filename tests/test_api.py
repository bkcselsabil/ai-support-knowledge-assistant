from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_question():
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422


def test_chat_returns_agent_answer(monkeypatch):
    from app import main

    def fake_run_agent(question, history=None):
        return {
            "answer": "Employees receive 30 vacation days.",
            "sources": [
                {
                    "title": "Vacation Policy",
                    "document": "confluence_917505.txt",
                    "url": None,
                    "source": "confluence",
                }
            ],
            "messages": [
                {"role": "user", "content": question},
                {
                    "role": "assistant",
                    "content": "Employees receive 30 vacation days.",
                },
            ],
        }

    monkeypatch.setattr(main, "run_agent", fake_run_agent)

    response = client.post(
        "/chat",
        json={"question": "How many vacation days do employees receive?"},
    )

    assert response.status_code == 200
    assert response.json() == {
        "answer": "Employees receive 30 vacation days.",
        "sources": [
            {
                "title": "Vacation Policy",
                "document": "confluence_917505.txt",
                "url": None,
                "source": "confluence",
            }
        ],
        "conversation_id": response.json()["conversation_id"],
    }

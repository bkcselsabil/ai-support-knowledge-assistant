from fastapi.testclient import TestClient

from app.main import app
from app.auth import require_auth



def override_auth():
    return {
        "sub": "test-user",
        "org_id": "local-dev",
    }


def test_health_check(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_chat_rejects_empty_question(client):
    response = client.post("/chat", json={"question": ""})

    assert response.status_code == 422


def test_chat_returns_agent_answer(client, db_session):
    from db.models import Conversation, Message, MessageSource

    response = client.post(
        "/chat",
        json={
            "question": "How many vacation days do employees receive?"
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["answer"] == "Full-time employees receive 30 vacation days."
    assert data["sources"][0]["title"] == "Vacation Policy"
    assert data["conversation_id"]

    conversation = db_session.get(
        Conversation,
        data["conversation_id"],
    )

    assert conversation is not None
    assert conversation.user_id == "test-user-1"

    messages = (
        db_session.query(Message)
        .filter(Message.conversation_id == data["conversation_id"])
        .order_by(Message.id)
        .all()
    )

    assert len(messages) == 2
    assert messages[0].role == "user"
    assert messages[1].role == "assistant"

    sources = (
        db_session.query(MessageSource)
        .filter(MessageSource.message_id == messages[1].id)
        .all()
    )

    assert len(sources) == 1
    assert sources[0].title == "Vacation Policy"


def test_user_can_continue_own_conversation(
    client, db_session, monkeypatch
):
    from app import main
    from db.models import Conversation, Message

    conversation = Conversation(
        user_id="test-user-1",
        tenant_id="local-dev",
    )
    db_session.add(conversation)
    db_session.flush()

    db_session.add(
        Message(
            conversation_id=conversation.id,
            role="user",
            content="How many vacation days do employees get?",
        )
    )
    db_session.add(
        Message(
            conversation_id=conversation.id,
            role="assistant",
            content="Full-time employees receive 30 vacation days.",
        )
    )
    db_session.commit()

    def fake_invoke(state):
        assert state["message"] == "Explain that more simply."

        assert state["history"] == [
            {
                "role": "user",
                "content": "How many vacation days do employees get?",
            },
            {
                "role": "assistant",
                "content": "Full-time employees receive 30 vacation days.",
            },
        ]

        return {
            "result": "You get 30 vacation days each year.",
            "sources": [],
        }

    monkeypatch.setattr(main.langgraph_app, "invoke", fake_invoke)

    response = client.post(
        "/chat",
        json={
            "question": "Explain that more simply.",
            "conversation_id": conversation.id,
        },
    )

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "You get 30 vacation days each year."
    )
    assert response.json()["conversation_id"] == conversation.id

    messages = (
        db_session.query(Message)
        .filter(Message.conversation_id == conversation.id)
        .order_by(Message.id)
        .all()
    )

    assert len(messages) == 4
    assert messages[-1].role == "assistant"
    assert messages[-1].content == (
        "You get 30 vacation days each year."
    )


def test_user_cannot_access_another_users_conversation(
    client, db_session
):
    from db.models import Conversation

    conversation = Conversation(
        user_id="another-user",
        tenant_id="local-dev",
    )

    db_session.add(conversation)
    db_session.commit()

    response = client.post(
        "/chat",
        json={
            "question": "Show me the private messages",
            "conversation_id": conversation.id,
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "You do not have access to this conversation."
    )


def test_chat_rejects_user_without_id(client):
    def override_auth_without_user_id():
        return {"org_id": "local-dev"}

    original_override = app.dependency_overrides.get(require_auth)

    app.dependency_overrides[require_auth] = (
        override_auth_without_user_id
    )

    try:
        response = client.post(
            "/chat",
            json={"question": "Hello"},
        )

        assert response.status_code == 401
        assert response.json()["detail"] == (
            "Authenticated user ID is missing."
        )

    finally:
        if original_override is None:
            app.dependency_overrides.pop(require_auth, None)
        else:
            app.dependency_overrides[require_auth] = original_override
def test_chat_rejects_user_without_organization(client):
    def override_auth_without_organization():
        return {"sub": "test-user-1"}

    original_override = app.dependency_overrides.get(require_auth)

    app.dependency_overrides[require_auth] = (
        override_auth_without_organization
    )

    try:
        response = client.post(
            "/chat",
            json={"question": "Hello"},
        )

        assert response.status_code == 403
        assert response.json()["detail"] == (
            "An active organization is required."
        )

    finally:
        if original_override is None:
            app.dependency_overrides.pop(require_auth, None)
        else:
            app.dependency_overrides[require_auth] = original_override

def test_user_cannot_access_another_tenants_conversation(
    client, db_session
):
    from db.models import Conversation

    conversation = Conversation(
        user_id="test-user-1",
        tenant_id="organization-B",
    )

    db_session.add(conversation)
    db_session.commit()

    response = client.post(
        "/chat",
        json={
            "question": "Show me the private conversation",
            "conversation_id": conversation.id,
        },
    )

    assert response.status_code == 403
    assert response.json()["detail"] == (
        "You do not have access to this conversation."
    )
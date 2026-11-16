import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from db.database import DATABASE_URL
from db.models import Base

from fastapi.testclient import TestClient

from app.main import app
from app.auth import require_auth
from db.database import get_db

TEST_DATABASE_URL = DATABASE_URL.set(
    database="ai_support_test_db"
)

test_engine = create_engine(TEST_DATABASE_URL)

TestSessionLocal = sessionmaker(
    bind=test_engine,
    autoflush=False,
    expire_on_commit=False,
)


@pytest.fixture
def db_session():
    Base.metadata.create_all(bind=test_engine)

    connection = test_engine.connect()
    transaction = connection.begin()

    session = TestSessionLocal(
        bind=connection,
        join_transaction_mode="create_savepoint",
    )

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()
        
@pytest.fixture
def client(db_session, monkeypatch):
    def override_get_db():
        yield db_session

    def override_auth():
        return {"sub": "test-user-1",
                "org_id": "local-dev",}

    def fake_langgraph_invoke(state):
        return {
            "result": "Full-time employees receive 30 vacation days.",
            "sources": [
                {
                    "title": "Vacation Policy",
                    "document": "confluence_917505.txt",
                    "url": None,
                    "source": "confluence",
                }
            ],
        }

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[require_auth] = override_auth

    monkeypatch.setattr(
        "app.main.langgraph_app.invoke",
        fake_langgraph_invoke,
    )

    try:
        with TestClient(app) as test_client:
            yield test_client
    finally:
        app.dependency_overrides.clear()
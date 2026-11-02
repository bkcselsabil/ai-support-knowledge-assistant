from pathlib import Path
from uuid import uuid4

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.auth import require_auth
from agent import run_agent


BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app = FastAPI(
    title="AI Support Knowledge Assistant",
    description="API for answering support questions using RAG and an AI agent.",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


class Source(BaseModel):
    title: str
    document: str
    url: str | None = None
    source: str = "local"


class ChatResponse(BaseModel):
    answer: str
    sources: list[Source] = Field(default_factory=list)
    conversation_id: str


# Temporary in-memory conversation storage.
# Each conversation is associated with its owner.
# All conversations are lost when the server restarts.
conversations: dict[str, dict] = {}


@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    user: dict = Depends(require_auth),
):
    user_id = user.get("sub")

    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authenticated user ID is missing.",
        )

    conversation_id = request.conversation_id or str(uuid4())

    conversation = conversations.get(conversation_id)

    if conversation is not None and conversation["user_id"] != user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this conversation.",
        )

    history = conversation["messages"] if conversation else []

    result = run_agent(
        request.question,
        history=history,
        tenant_id="local-dev",
    )

    conversations[conversation_id] = {
        "user_id": user_id,
        "messages": result["messages"],
    }

    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=conversation_id,
    )
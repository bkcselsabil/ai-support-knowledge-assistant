from pathlib import Path
from uuid import uuid4
from sqlalchemy.orm import Session

from db.database import get_db
from db.models import Conversation, Message, MessageSource
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from examples.langgraph_test import app as langgraph_app
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
    db: Session = Depends(get_db),
):
    user_id = user.get("sub")
    tenant_id = user.get("org_id")

    if not tenant_id:
        raise HTTPException(
            status_code=403,
            detail="An active organization is required.",
        )
    if not user_id:
        raise HTTPException(
            status_code=401,
            detail="Authenticated user ID is missing.",
        )

    conversation_id = request.conversation_id or str(uuid4())

    conversation = db.get(Conversation, conversation_id)

    if conversation is not None and conversation.user_id != user_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this conversation.",
        )

    if conversation is not None and conversation.tenant_id != tenant_id:
        raise HTTPException(
            status_code=403,
            detail="You do not have access to this conversation.",
        )

    history = []

    if conversation is not None:
        messages = (
            db.query(Message)
            .filter(Message.conversation_id == conversation_id)
            .order_by(Message.id)
            .all()
        )

        history = [
            {"role": message.role, "content": message.content}
            for message in messages
        ]

    langgraph_history = to_langgraph_history(history)
    result = langgraph_app.invoke({
        "message": request.question,
        "result": "",
        "route": "",
        "history": langgraph_history,
        "tenant_id": tenant_id,
    })

    current_sources = result.get("sources", [])
    current_sources = result.get("sources", [])

    if conversation is None:
        conversation = Conversation(
            id=conversation_id,
            user_id=user_id,
            tenant_id=tenant_id,
        )
        db.add(conversation)

    user_message = Message(
        conversation_id=conversation_id,
        role="user",
        content=request.question,
    )
    db.add(user_message)

    assistant_message = Message(
        conversation_id=conversation_id,
        role="assistant",
        content=result["result"],
    )
    db.add(assistant_message)

    db.flush()

    for source in current_sources:
        db.add(
            MessageSource(
                message_id=assistant_message.id,
                title=source.get("title") or "Unknown",
                document=source.get("document") or "Unknown",
                url=source.get("url"),
                source=source.get("source") or "unknown",
            )
        )

    db.commit()

    return ChatResponse(
        answer=result["result"],
        sources=current_sources,
        conversation_id=conversation_id,
    )
        
def to_langgraph_history(messages: list[dict]) -> list[dict]:
    history = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role not in ("user", "assistant"):
            continue

        if isinstance(content, str):
            text = content

        elif isinstance(content, list):
            text_parts = []

            for block in content:
                if isinstance(block, dict):
                    if block.get("type") == "text":
                        text_parts.append(block.get("text", ""))

                elif getattr(block, "type", None) == "text":
                    text_parts.append(getattr(block, "text", ""))

            text = "\n".join(text_parts)

        else:
            continue

        if text.strip():
            history.append({
                "role": role,
                "content": text,
            })

    return history

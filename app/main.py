from pathlib import Path
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from agent import run_agent

app = FastAPI(
    title="AI Support Knowledge Assistant",
    description="API for answering support questions using RAG and an AI agent.",
    version="0.1.0",
)
BASE_DIR = Path(__file__).resolve().parent.parent
FRONTEND_DIR = BASE_DIR / "frontend"

app.mount(
    "/static",
    StaticFiles(directory=FRONTEND_DIR),
    name="static",
)

@app.get("/")
async def serve_frontend():
    return FileResponse(FRONTEND_DIR / "index.html")
conversations = {}
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://127.0.0.1:5500",
        "http://localhost:5500",
    ],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
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
    sources: list[Source] = []
    conversation_id: str

@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    conversation_id = request.conversation_id or str(uuid.uuid4())

    history = conversations.get(conversation_id, [])

    result = run_agent(request.question, history=history)

    conversations[conversation_id] = result["messages"]
    return ChatResponse(
        answer=result["answer"],
        sources=result["sources"],
        conversation_id=conversation_id,
    )

"""STEP 9: The API that Vercel runs.
Local test:  uvicorn app:app --reload   ->  open http://127.0.0.1:8000
"""
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from rag import IronStoreAssistant

app = FastAPI(title="IronStore Enterprise AI Assistant")

_bot = None


def get_bot() -> IronStoreAssistant:
    """Create the assistant once and reuse it (connecting to Pinecone/OpenAI takes time)."""
    global _bot
    if _bot is None:
        _bot = IronStoreAssistant()
    return _bot


class Message(BaseModel):
    role: str      # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    question: str
    history: list[Message] = []


@app.get("/", response_class=HTMLResponse)
def home():
    """Send the chat web page."""
    return (Path(__file__).parent / "templates" / "index.html").read_text(encoding="utf-8")


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.post("/api/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Please type a question.")
    if len(question) > 2000:
        raise HTTPException(status_code=400, detail="Question is too long (max 2000 characters).")
    history = [m.model_dump() for m in req.history][-10:]   # last 10 messages are enough
    try:
        return get_bot().ask(question, history)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Assistant error: {e}")

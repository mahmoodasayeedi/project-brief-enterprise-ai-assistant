"""API tests.   Run:  pytest -v
These call the real assistant (OpenAI + Pinecone), so they cost a few cents."""
import json

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_chat_page_loads():
    r = client.get("/")
    assert r.status_code == 200
    assert "IronStore" in r.text


def test_empty_question_is_rejected():
    r = client.post("/api/chat", json={"question": "   "})
    assert r.status_code == 400


def test_answer_has_sources_with_page_and_link():
    r = client.post("/api/chat", json={"question": "How do I request annual leave?"})
    assert r.status_code == 200
    data = r.json()
    assert "peoplehub" in data["answer"].lower()
    src = data["sources"][0]
    assert src["page"] and src["url"].startswith("/files/")
    assert client.get(src["url"].split("#")[0]).status_code == 200   # the linked PDF really exists


def test_off_topic_question_is_declined():
    r = client.post("/api/chat", json={"question": "What is the capital of France?"})
    assert r.status_code == 200
    assert r.json()["sources"] == []


def test_streaming_sends_meta_then_tokens():
    with client.stream("POST", "/api/chat/stream", json={"question": "How do I request annual leave?"}) as r:
        events = [json.loads(line) for line in r.iter_lines() if line]
    assert events[0]["type"] == "meta"
    assert any(e["type"] == "token" for e in events)

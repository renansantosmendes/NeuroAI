import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
from src.api.app import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@patch("src.api.app.get_rag")
def test_query_basic(mock_get_rag):
    mock_chain = MagicMock()
    mock_chain.query.return_value = {
        "question": "What is RAG?",
        "answer": "Retrieval Augmented Generation",
        "context": []
    }
    mock_get_rag.return_value = (mock_chain, MagicMock(), MagicMock())
    
    response = client.post("/query", json={"question": "What is RAG?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Retrieval Augmented Generation"

@patch("src.api.app.get_rag")
def test_query_graph(mock_get_rag):
    mock_graph = MagicMock()
    mock_graph.query.return_value = {
        "question": "What is RAG?",
        "answer": "Graph Answer",
        "context": []
    }
    mock_get_rag.return_value = (MagicMock(), mock_graph, MagicMock())
    
    response = client.post("/query/graph", json={"question": "What is RAG?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Graph Answer"

@patch("src.api.app.get_rag")
def test_query_advanced(mock_get_rag):
    mock_adv = MagicMock()
    mock_adv.query.return_value = {
        "question": "What is RAG?",
        "answer": "Advanced Answer",
        "context": [],
        "needs_retrieval": True,
        "confidence": 0.9
    }
    mock_get_rag.return_value = (MagicMock(), MagicMock(), mock_adv)
    
    response = client.post("/query/advanced", json={"question": "What is RAG?"})
    assert response.status_code == 200
    assert response.json()["answer"] == "Advanced Answer"
    assert response.json()["needs_retrieval"] is True

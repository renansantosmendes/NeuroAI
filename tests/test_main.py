from unittest.mock import patch, MagicMock
from src.main import get_sample_documents, run_rag_demo

def test_get_sample_documents():
    docs = get_sample_documents()
    assert len(docs) > 0
    assert "Machine Learning" in docs[0].page_content

@patch("src.main.Config")
@patch("src.main.PineconeManager")
@patch("src.main.RAGChain")
@patch("src.main.RAGGraph")
@patch("src.main.AdvancedRAGGraph")
def test_run_rag_demo(mock_adv, mock_graph, mock_chain, mock_pm, mock_config):
    # Mocking component returns to simulate demo flow
    mock_pm.return_value.similarity_search.return_value = []
    mock_chain.return_value.query.return_value = {"answer": "ans"}
    mock_graph.return_value.query.return_value = {"answer": "ans"}
    mock_adv.return_value.query.return_value = {"answer": "ans", "confidence": 0.9, "needs_retrieval": True}
    
    # Just verify it runs without error when components are mocked
    run_rag_demo()
    mock_config.validate.assert_called_once()

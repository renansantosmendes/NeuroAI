from unittest.mock import patch, MagicMock
from src.core.demo import get_sample_documents, run_rag_demo

def test_get_sample_documents():
    docs = get_sample_documents()
    assert len(docs) > 0
    assert "Machine Learning" in docs[0].page_content

@patch("src.core.demo.Config")
@patch("src.core.demo.PineconeManager")
@patch("src.core.demo.RAGChain")
@patch("src.core.demo.RAGGraph")
@patch("src.core.demo.AdvancedRAGGraph")
def test_run_rag_demo(mock_adv, mock_graph, mock_chain, mock_pm, mock_config):
    # Mocking component returns to simulate demo flow
    mock_pm.return_value.similarity_search.return_value = []
    mock_chain.return_value.query.return_value = {"answer": "ans"}
    mock_graph.return_value.query.return_value = {"answer": "ans"}
    mock_adv.return_value.query.return_value = {"answer": "ans", "confidence": 0.9, "needs_retrieval": True}
    
    # Just verify it runs without error when components are mocked
    run_rag_demo()
    mock_config.validate.assert_called_once()

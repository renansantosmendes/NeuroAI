import os
from unittest.mock import MagicMock, patch
from src.core.chain import RAGChain
from langchain_core.documents import Document

os.environ["OPENAI_API_KEY"] = "fake-key"
os.environ["PINECONE_API_KEY"] = "fake-key"

def test_rag_chain_format_context():
    pm = MagicMock()
    chain = RAGChain(pm)
    docs = [Document(page_content="doc1"), Document(page_content="doc2")]
    formatted = chain.format_context(docs)
    assert "doc1" in formatted
    assert "doc2" in formatted
    assert "---" in formatted

@patch("src.core.chain.ChatOpenAI")
def test_rag_chain_query(mock_llm_class):
    pm = MagicMock()
    pm.similarity_search.return_value = [Document(page_content="context")]
    
    chain = RAGChain(pm)
    
    # Mock the generate method to avoid testing internal LCEL
    with patch.object(RAGChain, 'generate', return_value="mocked answer"):
        result = chain.query("test question")
        assert result["answer"] == "mocked answer"
        assert result["question"] == "test question"
        assert len(result["context"]) == 1

@patch("src.core.chain.ChatOpenAI")
def test_rag_chain_generate(mock_llm_class):
    pm = MagicMock()
    chain = RAGChain(pm)
    
    mock_runnable = MagicMock()
    mock_runnable.invoke.return_value = "final answer"
    
    with patch.object(RAGChain, '_get_chain', return_value=mock_runnable):
        ans = chain.generate("q", [Document(page_content="c")])
        assert ans == "final answer"

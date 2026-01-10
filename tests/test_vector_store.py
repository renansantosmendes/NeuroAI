import pytest
from unittest.mock import patch, MagicMock
from src.core.vector_store import PineconeManager
from langchain_core.documents import Document

@pytest.fixture
def mock_pinecone_manager():
    with patch("src.core.vector_store.Pinecone") as mock_pc, \
         patch("src.core.vector_store.OpenAIEmbeddings") as mock_emb, \
         patch("src.core.vector_store.ServerlessSpec"):
        pm = PineconeManager("p-key", "o-key", "t-index")
        yield pm, mock_pc, mock_emb

def test_create_index(mock_pinecone_manager):
    pm, mock_pc, _ = mock_pinecone_manager
    mock_instance = mock_pc.return_value
    mock_instance.list_indexes.return_value = []
    
    # Mock describe_index to return a status that eventually becomes ready
    mock_status = MagicMock()
    mock_status.status = {"ready": True}
    mock_instance.describe_index.return_value = mock_status
    
    pm.create_index()
    mock_instance.create_index.assert_called_once()

def test_create_from_documents(mock_pinecone_manager):
    pm, mock_pc, _ = mock_pinecone_manager
    mock_instance = mock_pc.return_value
    mock_instance.list_indexes.return_value = [MagicMock(name="t-index")]
    
    with patch("src.core.vector_store.PineconeVectorStore") as mock_vs:
        docs = [Document(page_content="test")]
        pm.create_from_documents(docs)
        mock_vs.return_value.add_documents.assert_called_once()

def test_similarity_search(mock_pinecone_manager):
    pm, _, _ = mock_pinecone_manager
    pm.vector_store = MagicMock()
    pm.vector_store.similarity_search.return_value = [Document(page_content="res")]
    
    results = pm.similarity_search("query")
    assert results[0].page_content == "res"

def test_delete_index(mock_pinecone_manager):
    pm, mock_pc, _ = mock_pinecone_manager
    mock_instance = mock_pc.return_value
    # Use a mock object that has a 'name' attribute
    mock_idx = MagicMock()
    mock_idx.name = "t-index"
    mock_instance.list_indexes.return_value = [mock_idx]
    
    pm.delete_index()
    mock_instance.delete_index.assert_called_with("t-index")

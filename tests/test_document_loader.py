from unittest.mock import patch, MagicMock
from src.core.document_loader import load_documents, split_documents
from langchain_core.documents import Document

def test_load_documents_text():
    with patch("src.core.document_loader.TextLoader") as mock_loader:
        mock_instance = mock_loader.return_value
        mock_instance.load.return_value = [Document(page_content="test")]
        
        docs = load_documents("test.txt", "text")
        assert len(docs) == 1
        assert docs[0].page_content == "test"

def test_load_documents_web():
    with patch("src.core.document_loader.WebBaseLoader") as mock_loader:
        mock_instance = mock_loader.return_value
        mock_instance.load.return_value = [Document(page_content="web test")]
        
        docs = load_documents("https://example.com", "web")
        assert len(docs) == 1
        assert docs[0].page_content == "web test"

def test_split_documents():
    docs = [Document(page_content="This is a long sentence that should be split.")]
    chunks = split_documents(docs, chunk_size=10, chunk_overlap=0)
    assert len(chunks) > 1

from typing import TypedDict, List
from langchain_core.documents import Document

class RAGState(TypedDict):
    """
    State object that flows through the RAG graph.
    """
    question: str
    context: List[Document]
    answer: str

class AdvancedRAGState(TypedDict):
    """
    Extended state for advanced RAG with routing capabilities.
    """
    question: str
    context: List[Document]
    answer: str
    needs_retrieval: bool
    confidence: float

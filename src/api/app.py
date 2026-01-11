import logging
from fastapi import FastAPI, HTTPException, Depends
from typing import List
from src.config import Config
from src.core.vector_store import PineconeManager
from src.core.chain import RAGChain
from src.core.graph import RAGGraph, AdvancedRAGGraph
from src.api.schemas import QueryRequest, QueryResponse, AdvancedQueryResponse, HealthResponse, DocumentSchema

logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NeuroAI RAG API",
    description="API for interacting with the Retrieval-Augmented Generation agent.",
    version="1.0.0"
)

# Global instances (simplified for demo)
# In a production app, these would be managed via dependency injection or a startup event
_pinecone_manager = None
_rag_chain = None
_rag_graph = None
_advanced_rag = None

def get_rag():
    global _pinecone_manager, _rag_chain, _rag_graph, _advanced_rag
    if _pinecone_manager is None:
        Config.validate()
        _pinecone_manager = PineconeManager(
            pinecone_api_key=Config.PINECONE_API_KEY,
            openai_api_key=Config.OPENAI_API_KEY,
            index_name=Config.INDEX_NAME
        )
        # Assuming index and data are already set up for this demo
        # If not, you might want to call connect_to_existing()
        _pinecone_manager.connect_to_existing()
        _rag_chain = RAGChain(_pinecone_manager)
        _rag_graph = RAGGraph(_rag_chain)
        _advanced_rag = AdvancedRAGGraph(_rag_chain)
    return _rag_chain, _rag_graph, _advanced_rag

@app.get("/health", response_model=HealthResponse, tags=["Utility"])
async def health_check():
    """Check the health status of the API."""
    return {"status": "healthy", "version": "1.0.0"}

@app.post("/query", response_model=QueryResponse, tags=["RAG"])
async def query_basic(request: QueryRequest):
    """
    Execute a basic RAG query.
    Retrieves context and generates an answer using the standard chain.
    """
    logger.info(f"API: Received basic query. Question: {request.question}")
    try:
        chain, _, _ = get_rag()
        result = chain.query(request.question, k=request.k)
        
        # Format response
        formatted_context = [
            DocumentSchema(page_content=doc.page_content, metadata=doc.metadata) 
            for doc in result["context"]
        ]
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context": formatted_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/graph", response_model=QueryResponse, tags=["RAG"])
async def query_graph(request: QueryRequest):
    """
    Execute a RAG query using the LangGraph stateful workflow.
    """
    logger.info(f"API: Received graph query. Question: {request.question}")
    try:
        _, graph, _ = get_rag()
        result = graph.query(request.question)
        
        formatted_context = [
            DocumentSchema(page_content=doc.page_content, metadata=doc.metadata) 
            for doc in result["context"]
        ]
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context": formatted_context
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/query/advanced", response_model=AdvancedQueryResponse, tags=["RAG"])
async def query_advanced(request: QueryRequest):
    """
    Execute an advanced RAG query with routing.
    Determines if retrieval is needed based on the query type.
    """
    logger.info(f"API: Received advanced query. Question: {request.question}")
    try:
        _, _, adv_graph = get_rag()
        result = adv_graph.query(request.question)
        
        formatted_context = [
            DocumentSchema(page_content=doc.page_content, metadata=doc.metadata or {}) 
            for doc in result["context"]
        ]
        return {
            "question": result["question"],
            "answer": result["answer"],
            "context": formatted_context,
            "needs_retrieval": result["needs_retrieval"],
            "confidence": result["confidence"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

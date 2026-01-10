from langchain_core.documents import Document
from src.config import Config
from src.core.vector_store import PineconeManager
from src.core.chain import RAGChain
from src.core.graph import RAGGraph, AdvancedRAGGraph

def get_sample_documents():
    sample_texts = [
        "Machine Learning is a subset of artificial intelligence.",
        "RAG combines retrieval systems with generative AI.",
        "Pinecone is a vector database for ML applications.",
        "LangGraph is for building stateful applications with LLMs."
    ]
    return [Document(page_content=text) for text in sample_texts]

def run_rag_demo():
    # Validate configuration
    Config.validate()
    
    # Initialize components
    pm = PineconeManager(
        pinecone_api_key=Config.PINECONE_API_KEY,
        openai_api_key=Config.OPENAI_API_KEY,
        index_name=Config.INDEX_NAME
    )
    
    # Setup knowledge base
    docs = get_sample_documents()
    pm.create_from_documents(docs)
    
    # Initialize Chain and Graphs
    rag_chain = RAGChain(pm)
    rag_graph = RAGGraph(rag_chain)
    advanced_rag = AdvancedRAGGraph(rag_chain)
    
    # Test queries
    print("\n--- Basic RAG Query ---")
    result = rag_chain.query("What is RAG?")
    print(f"Answer: {result['answer']}")
    
    print("\n--- LangGraph RAG Query ---")
    result = rag_graph.query("What is Machine Learning?")
    print(f"Answer: {result['answer']}")
    
    print("\n--- Advanced RAG Query (Document search) ---")
    result = advanced_rag.query("What is Pinecone?")
    print(f"Answer: {result['answer']}")
    print(f"Confidence: {result['confidence']}")
    
    print("\n--- Advanced RAG Query (General question) ---")
    result = advanced_rag.query("Hi, how are you?")
    print(f"Answer: {result['answer']}")
    print(f"Needs retrieval: {result['needs_retrieval']}")

if __name__ == "__main__":
    run_rag_demo()

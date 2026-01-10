"""
Basic RAG (Retrieval-Augmented Generation) Example using LangChain and LangGraph.

This module demonstrates a simple RAG pipeline that:
1. Loads and splits documents
2. Creates vector embeddings and stores them in Pinecone
3. Retrieves relevant context based on user queries
4. Generates answers using an LLM with the retrieved context
"""

import os
import time
from typing import TypedDict, List
from uuid import uuid4

from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from pinecone import Pinecone, ServerlessSpec

from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

load_dotenv()

# =============================================================================
# State Definition
# =============================================================================

class RAGState(TypedDict):
    """
    State object that flows through the RAG graph.
    
    Attributes:
        question: The user's input question.
        context: List of retrieved document chunks relevant to the question.
        answer: The generated answer based on the context.
    """
    question: str
    context: List[Document]
    answer: str


# =============================================================================
# Document Processing
# =============================================================================

def load_documents(source: str, source_type: str = "text"):
    """
    Load documents from a file or web URL.
    
    Args:
        source: Path to the file or URL to load.
        source_type: Type of source - 'text' for local files, 'web' for URLs.
    
    Returns:
        List of loaded Document objects.
    
    Example:
        >>> docs = load_documents("data.txt", "text")
        >>> docs = load_documents("https://example.com", "web")
    """
    if source_type == "web":
        loader = WebBaseLoader(source)
    else:
        loader = TextLoader(source)
    
    return loader.load()


def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Split documents into smaller chunks for better retrieval.
    
    Args:
        documents: List of Document objects to split.
        chunk_size: Maximum size of each chunk in characters.
        chunk_overlap: Number of overlapping characters between chunks.
    
    Returns:
        List of Document chunks.
    
    Example:
        >>> chunks = split_documents(docs, chunk_size=1000, chunk_overlap=100)
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)


# =============================================================================
# Pinecone Vector Store
# =============================================================================

class PineconeManager:
    """
    Manages the creation and querying of the Pinecone vector store.
    
    This class handles embedding generation, index management,
    and similarity search using Pinecone as the vector database.
    
    Attributes:
        embeddings: The embedding model used to convert text to vectors.
        pc: The Pinecone client instance.
        index_name: Name of the Pinecone index.
        vector_store: The PineconeVectorStore instance.
    """
    
    def __init__(self, pinecone_api_key: str = None, openai_api_key: str = None, index_name: str = "rag-index"):
        """
        Initialize the PineconeManager.
        
        Args:
            pinecone_api_key: Pinecone API key. If None, uses PINECONE_API_KEY env variable.
            openai_api_key: OpenAI API key. If None, uses OPENAI_API_KEY env variable.
            index_name: Name for the Pinecone index.
        """
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.index_name = index_name
        
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.vector_store = None
    
    def create_index(self, dimension: int = 1536, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1"):
        """
        Create a new Pinecone index if it doesn't exist.
        
        Args:
            dimension: Vector dimension (1536 for OpenAI text-embedding-ada-002).
            metric: Distance metric - 'cosine', 'euclidean', or 'dotproduct'.
            cloud: Cloud provider for serverless - 'aws', 'gcp', or 'azure'.
            region: Cloud region for the index.
        
        Returns:
            The Pinecone index object.
        
        Example:
            >>> manager = PineconeManager()
            >>> manager.create_index(dimension=1536)
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(
                    cloud=cloud,
                    region=region
                )
            )
            # Wait for index to be ready
            while not self.pc.describe_index(self.index_name).status["ready"]:
                print("Waiting for index to be ready...")
                time.sleep(1)
            print("Index created and ready!")
        else:
            print(f"Using existing index: {self.index_name}")
        
        return self.pc.Index(self.index_name)
    
    def create_from_documents(self, documents: List[Document], namespace: str = "default"):
        """
        Create a vector store from a list of documents.
        
        Args:
            documents: List of Document objects to index.
            namespace: Pinecone namespace for organizing vectors.
        
        Returns:
            The PineconeVectorStore instance.
        
        Example:
            >>> manager = PineconeManager()
            >>> manager.create_index()
            >>> manager.create_from_documents(chunks)
        """
        # Ensure index exists
        self.create_index()
        
        # Generate unique IDs for documents
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        # Create vector store and add documents
        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings,
            namespace=namespace
        )
        
        self.vector_store.add_documents(documents=documents, ids=uuids)
        print(f"Added {len(documents)} documents to Pinecone")
        
        return self.vector_store
    
    def connect_to_existing(self, namespace: str = "default"):
        """
        Connect to an existing Pinecone index.
        
        Args:
            namespace: Pinecone namespace to use.
        
        Returns:
            The PineconeVectorStore instance.
        
        Example:
            >>> manager = PineconeManager(index_name="my-existing-index")
            >>> manager.connect_to_existing()
        """
        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings,
            namespace=namespace
        )
        
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 3, namespace: str = "default"):
        """
        Perform similarity search on the vector store.
        
        Args:
            query: The search query string.
            k: Number of similar documents to retrieve.
            namespace: Pinecone namespace to search in.
        
        Returns:
            List of the k most similar Document objects.
        
        Raises:
            ValueError: If vector store has not been initialized.
        
        Example:
            >>> results = manager.similarity_search("What is RAG?", k=5)
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_from_documents or connect_to_existing first.")
        
        return self.vector_store.similarity_search(query, k=k, namespace=namespace)
    
    def similarity_search_with_score(self, query: str, k: int = 3, namespace: str = "default"):
        """
        Perform similarity search and return documents with scores.
        
        Args:
            query: The search query string.
            k: Number of similar documents to retrieve.
            namespace: Pinecone namespace to search in.
        
        Returns:
            List of tuples (Document, score).
        
        Example:
            >>> results = manager.similarity_search_with_score("What is RAG?")
            >>> for doc, score in results:
            ...     print(f"Score: {score}, Content: {doc.page_content[:50]}...")
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        return self.vector_store.similarity_search_with_score(query, k=k, namespace=namespace)
    
    def delete_index(self):
        """
        Delete the Pinecone index.
        
        Warning: This permanently deletes all vectors in the index.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name in existing_indexes:
            self.pc.delete_index(self.index_name)
            print(f"Deleted index: {self.index_name}")
        else:
            print(f"Index {self.index_name} does not exist")
    
    def clear_namespace(self, namespace: str = "default"):
        """
        Clear all vectors in a specific namespace.
        
        Args:
            namespace: The namespace to clear.
        """
        index = self.pc.Index(self.index_name)
        index.delete(delete_all=True, namespace=namespace)
        print(f"Cleared namespace: {namespace}")
    
    def get_index_stats(self):
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dictionary with index statistics.
        """
        index = self.pc.Index(self.index_name)
        return index.describe_index_stats()


# =============================================================================
# RAG Chain Components
# =============================================================================

class RAGChain:
    """
    Implements the RAG chain for question answering.
    
    This class combines retrieval and generation to answer
    questions based on the provided document context.
    
    Attributes:
        pinecone_manager: PineconeManager for document retrieval.
        llm: The language model for answer generation.
        prompt: The prompt template for the LLM.
    """
    
    def __init__(self, pinecone_manager: PineconeManager, model_name: str = "gpt-4o-mini"):
        """
        Initialize the RAG chain.
        
        Args:
            pinecone_manager: An initialized PineconeManager instance.
            model_name: Name of the OpenAI model to use.
        """
        self.pinecone_manager = pinecone_manager
        self.llm = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant that answers questions based on the provided context.
            
Context:
{context}

Question: {question}

Instructions:
- Answer the question based ONLY on the context provided above.
- If the context doesn't contain enough information to answer, say so.
- Be concise and direct in your response.

Answer:"""
        )
    
    def format_context(self, documents: List[Document]):
        """
        Format retrieved documents into a single context string.
        
        Args:
            documents: List of Document objects to format.
        
        Returns:
            Formatted string with all document contents.
        """
        return "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    def retrieve(self, question: str, k: int = 3):
        """
        Retrieve relevant documents for a question.
        
        Args:
            question: The user's question.
            k: Number of documents to retrieve.
        
        Returns:
            List of relevant Document objects.
        """
        return self.pinecone_manager.similarity_search(question, k=k)
    
    def retrieve_with_scores(self, question: str, k: int = 3):
        """
        Retrieve relevant documents with similarity scores.
        
        Args:
            question: The user's question.
            k: Number of documents to retrieve.
        
        Returns:
            List of tuples (Document, score).
        """
        return self.pinecone_manager.similarity_search_with_score(question, k=k)
    
    def generate(self, question: str, context: List[Document]):
        """
        Generate an answer using the LLM.
        
        Args:
            question: The user's question.
            context: List of relevant Document objects.
        
        Returns:
            The generated answer string.
        """
        formatted_context = self.format_context(context)
        
        chain = self.prompt | self.llm | StrOutputParser()
        
        return chain.invoke({
            "context": formatted_context,
            "question": question
        })
    
    def query(self, question: str, k: int = 3):
        """
        Execute the full RAG pipeline for a question.
        
        Args:
            question: The user's question.
            k: Number of documents to retrieve.
        
        Returns:
            Dictionary with question, context, and answer.
        
        Example:
            >>> result = rag_chain.query("What is machine learning?")
            >>> print(result["answer"])
        """
        context = self.retrieve(question, k=k)
        answer = self.generate(question, context)
        
        return {
            "question": question,
            "context": context,
            "answer": answer
        }


# =============================================================================
# LangGraph RAG Implementation
# =============================================================================

class RAGGraph:
    """
    RAG implementation using LangGraph for stateful workflow management.
    
    This class creates a graph-based RAG pipeline with explicit
    nodes for retrieval and generation steps.
    
    Attributes:
        rag_chain: The underlying RAGChain instance.
        graph: The compiled LangGraph workflow.
    """
    
    def __init__(self, rag_chain: RAGChain):
        """
        Initialize the RAG graph.
        
        Args:
            rag_chain: An initialized RAGChain instance.
        """
        self.rag_chain = rag_chain
        self.graph = self._build_graph()
    
    def _retrieve_node(self, state: RAGState):
        """
        Graph node for document retrieval.
        
        Args:
            state: Current graph state with the question.
        
        Returns:
            Updated state with retrieved context.
        """
        question = state["question"]
        context = self.rag_chain.retrieve(question)
        
        return {"context": context}
    
    def _generate_node(self, state: RAGState):
        """
        Graph node for answer generation.
        
        Args:
            state: Current graph state with question and context.
        
        Returns:
            Updated state with generated answer.
        """
        question = state["question"]
        context = state["context"]
        answer = self.rag_chain.generate(question, context)
        
        return {"answer": answer}
    
    def _build_graph(self):
        """
        Build the LangGraph workflow.
        
        Returns:
            Compiled graph ready for execution.
        """
        workflow = StateGraph(RAGState)
        
        # Add nodes
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        
        # Define edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        
        return workflow.compile()
    
    def query(self, question: str):
        """
        Execute the RAG graph for a question.
        
        Args:
            question: The user's question.
        
        Returns:
            Final state with question, context, and answer.
        
        Example:
            >>> result = rag_graph.query("Explain neural networks")
            >>> print(result["answer"])
        """
        initial_state = {
            "question": question,
            "context": [],
            "answer": ""
        }
        
        return self.graph.invoke(initial_state)


# =============================================================================
# Advanced RAG Graph with Routing
# =============================================================================

class AdvancedRAGState(TypedDict):
    """
    Extended state for advanced RAG with routing capabilities.
    
    Attributes:
        question: The user's input question.
        context: List of retrieved document chunks.
        answer: The generated answer.
        needs_retrieval: Flag indicating if retrieval is needed.
        confidence: Confidence score of the answer.
    """
    question: str
    context: List[Document]
    answer: str
    needs_retrieval: bool
    confidence: float


class AdvancedRAGGraph:
    """
    Advanced RAG implementation with conditional routing.
    
    This class extends the basic RAG graph with:
    - Query classification to determine if retrieval is needed
    - Confidence scoring for answers
    - Conditional routing based on query type
    """
    
    def __init__(self, rag_chain: RAGChain):
        """
        Initialize the advanced RAG graph.
        
        Args:
            rag_chain: An initialized RAGChain instance.
        """
        self.rag_chain = rag_chain
        self.classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.graph = self._build_graph()
    
    def _classify_query_node(self, state: AdvancedRAGState):
        """
        Classify if the query needs document retrieval.
        
        Args:
            state: Current graph state with the question.
        
        Returns:
            Updated state with needs_retrieval flag.
        """
        question = state["question"]
        
        classification_prompt = ChatPromptTemplate.from_template(
            """Analyze this question and determine if it requires searching through documents to answer.

Question: {question}

Respond with only 'YES' if the question needs document retrieval, or 'NO' if it's a general question."""
        )
        
        chain = classification_prompt | self.classifier_llm | StrOutputParser()
        result = chain.invoke({"question": question})
        
        needs_retrieval = "YES" in result.upper()
        
        return {"needs_retrieval": needs_retrieval}
    
    def _retrieve_node(self, state: AdvancedRAGState):
        """
        Retrieve relevant documents.
        
        Args:
            state: Current graph state.
        
        Returns:
            Updated state with retrieved context.
        """
        question = state["question"]
        context = self.rag_chain.retrieve(question)
        
        return {"context": context}
    
    def _generate_with_context_node(self, state: AdvancedRAGState):
        """
        Generate answer using retrieved context.
        
        Args:
            state: Current graph state with context.
        
        Returns:
            Updated state with answer and confidence.
        """
        question = state["question"]
        context = state["context"]
        answer = self.rag_chain.generate(question, context)
        
        # Simple confidence based on context availability
        confidence = 0.9 if len(context) >= 2 else 0.6
        
        return {"answer": answer, "confidence": confidence}
    
    def _generate_without_context_node(self, state: AdvancedRAGState):
        """
        Generate answer without document context.
        
        Args:
            state: Current graph state.
        
        Returns:
            Updated state with answer and confidence.
        """
        question = state["question"]
        
        prompt = ChatPromptTemplate.from_template(
            """Answer this general question concisely:

Question: {question}

Answer:"""
        )
        
        chain = prompt | self.rag_chain.llm | StrOutputParser()
        answer = chain.invoke({"question": question})
        
        return {"answer": answer, "confidence": 0.7, "context": []}
    
    def _route_query(self, state: AdvancedRAGState):
        """
        Route query based on classification.
        
        Args:
            state: Current graph state.
        
        Returns:
            Next node name to execute.
        """
        if state["needs_retrieval"]:
            return "retrieve"
        return "generate_direct"
    
    def _build_graph(self):
        """
        Build the advanced RAG graph with conditional routing.
        
        Returns:
            Compiled graph with routing logic.
        """
        workflow = StateGraph(AdvancedRAGState)
        
        # Add nodes
        workflow.add_node("classify", self._classify_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate_with_context", self._generate_with_context_node)
        workflow.add_node("generate_direct", self._generate_without_context_node)
        
        # Define edges
        workflow.set_entry_point("classify")
        workflow.add_conditional_edges(
            "classify",
            self._route_query,
            {
                "retrieve": "retrieve",
                "generate_direct": "generate_direct"
            }
        )
        workflow.add_edge("retrieve", "generate_with_context")
        workflow.add_edge("generate_with_context", END)
        workflow.add_edge("generate_direct", END)
        
        return workflow.compile()
    
    def query(self, question: str):
        """
        Execute the advanced RAG graph.
        
        Args:
            question: The user's question.
        
        Returns:
            Final state with answer and metadata.
        """
        initial_state = {
            "question": question,
            "context": [],
            "answer": "",
            "needs_retrieval": False,
            "confidence": 0.0
        }
        
        return self.graph.invoke(initial_state)


# =============================================================================
# Usage Example
# =============================================================================

def create_sample_documents():
    """
    Create sample documents for testing the RAG system.
    
    Returns:
        List of sample Document objects.
    """
    sample_texts = [
        """Machine Learning is a subset of artificial intelligence that enables 
        computers to learn from data without being explicitly programmed. 
        It uses algorithms to identify patterns and make decisions with minimal 
        human intervention.""",
        
        """Deep Learning is a type of machine learning based on artificial neural 
        networks. It uses multiple layers of processing to progressively extract 
        higher-level features from raw input data.""",
        
        """Natural Language Processing (NLP) is a branch of AI that helps computers 
        understand, interpret, and manipulate human language. It combines 
        computational linguistics with machine learning.""",
        
        """RAG (Retrieval-Augmented Generation) combines the power of retrieval 
        systems with generative AI. It first retrieves relevant documents from 
        a knowledge base, then uses them as context for generating accurate responses.""",
        
        """LangChain is a framework for developing applications powered by language 
        models. It provides tools for prompt management, chains, memory, and 
        integration with various data sources.""",
        
        """LangGraph is an extension of LangChain that enables building stateful, 
        multi-actor applications with LLMs. It uses a graph-based approach to 
        define complex workflows with conditional logic.""",
        
        """Pinecone is a vector database designed for machine learning applications.
        It provides fast and scalable similarity search, making it ideal for 
        RAG systems, recommendation engines, and semantic search applications.""",
        
        """Vector embeddings are numerical representations of data that capture 
        semantic meaning. They allow machines to understand relationships between 
        concepts by measuring distances in high-dimensional vector space."""
    ]
    
    return [Document(page_content=text) for text in sample_texts]


def main():
    """
    Main function demonstrating the RAG pipeline with Pinecone.
    
    This function shows how to:
    1. Create and process documents
    2. Build a Pinecone vector store
    3. Use both basic RAGChain and LangGraph implementations
    """
    print("=" * 60)
    print("RAG Example with LangChain, LangGraph and Pinecone")
    print("=" * 60)
    
    # Check for API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY environment variable not set.")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("\nError: PINECONE_API_KEY environment variable not set.")
        print("Please set it with: export PINECONE_API_KEY='your-key-here'")
        return
    
    # # Step 1: Create sample documents
    # print("\n1. Creating sample documents...")
    # documents = create_sample_documents()
    # print(f"   Created {len(documents)} documents")
    
    # # Step 2: Split documents
    # print("\n2. Splitting documents into chunks...")
    # chunks = split_documents(documents, chunk_size=300, chunk_overlap=30)
    # print(f"   Created {len(chunks)} chunks")
    
    # Step 3: Initialize Pinecone manager
    print("\n3. Initializing Pinecone...")
    pinecone_manager = PineconeManager(index_name="rag-langchain-example")
    pinecone_manager.connect_to_existing()
    # # Step 4: Create vector store with documents
    # print("\n4. Creating vector store and indexing documents...")
    # pinecone_manager.create_from_documents(chunks)
    
    # # Show index stats
    # stats = pinecone_manager.get_index_stats()
    # print(f"   Index stats: {stats}")
    
    # Step 5: Initialize RAG chain
    print("\n5. Initializing RAG chain...")
    rag_chain = RAGChain(pinecone_manager)
    print("   RAG chain ready")
    
    # Step 6: Test basic RAG chain
    print("\n6. Testing basic RAG chain...")
    question = "What is RAG and how does it work?"
    print(f"   Question: {question}")
    
    result = rag_chain.query(question)
    print(f"   Answer: {result['answer']}")
    print(f"   Retrieved {len(result['context'])} context documents")
    
    # Step 7: Test with scores
    print("\n7. Testing retrieval with similarity scores...")
    question2 = "What is Pinecone used for?"
    print(f"   Question: {question2}")
    
    results_with_scores = rag_chain.retrieve_with_scores(question2, k=3)
    for doc, score in results_with_scores:
        print(f"   Score: {score:.4f} - {doc.page_content[:60]}...")
    
    # Step 8: Test LangGraph RAG
    print("\n8. Testing LangGraph RAG...")
    rag_graph = RAGGraph(rag_chain)
    
    question3 = "How does LangGraph differ from LangChain?"
    print(f"   Question: {question3}")
    
    result3 = rag_graph.query(question3)
    print(f"   Answer: {result3['answer']}")
    
    # Step 9: Test Advanced RAG with routing
    print("\n9. Testing Advanced RAG with routing...")
    advanced_rag = AdvancedRAGGraph(rag_chain)
    
    # Query that needs retrieval
    question4 = "Explain deep learning and neural networks"
    print(f"\n   Question (needs retrieval): {question4}")
    result4 = advanced_rag.query(question4)
    print(f"   Needs Retrieval: {result4['needs_retrieval']}")
    print(f"   Confidence: {result4['confidence']}")
    print(f"   Answer: {result4['answer']}")
    
    # General query
    question5 = "What is 2 + 2?"
    print(f"\n   Question (general): {question5}")
    result5 = advanced_rag.query(question5)
    print(f"   Needs Retrieval: {result5['needs_retrieval']}")
    print(f"   Confidence: {result5['confidence']}")
    print(f"   Answer: {result5['answer']}")
    
    # Step 10: Cleanup (optional)
    print("\n10. Cleanup options:")
    print("    To delete the index, call: pinecone_manager.delete_index()")
    print("    To clear a namespace, call: pinecone_manager.clear_namespace('default')")
    
    print("\n" + "=" * 60)
    print("RAG Example Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
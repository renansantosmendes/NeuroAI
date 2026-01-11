import logging
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from .state import RAGState, AdvancedRAGState
from .chain import RAGChain

logger = logging.getLogger(__name__)

class RAGGraph:
    """
    RAG implementation using LangGraph for stateful workflow management.
    """
    
    def __init__(self, rag_chain: RAGChain):
        self.rag_chain = rag_chain
        self.graph = self._build_graph()
    
    def _retrieve_node(self, state: RAGState):
        question = state["question"]
        logger.info(f"RAGGraph: Node 'retrieve' using index: {self.rag_chain.pinecone_manager.index_name}")
        context = self.rag_chain.retrieve(question)
        return {"context": context}
    
    def _generate_node(self, state: RAGState):
        question = state["question"]
        context = state["context"]
        answer = self.rag_chain.generate(question, context)
        return {"answer": answer}
    
    def _build_graph(self):
        workflow = StateGraph(RAGState)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate", self._generate_node)
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)
        return workflow.compile()
    
    def query(self, question: str):
        initial_state = {
            "question": question,
            "context": [],
            "answer": ""
        }
        return self.graph.invoke(initial_state)

class AdvancedRAGGraph:
    """
    Advanced RAG implementation with conditional routing.
    """
    
    def __init__(self, rag_chain: RAGChain):
        self.rag_chain = rag_chain
        self.classifier_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.graph = self._build_graph()
    
    def _classify_query_node(self, state: AdvancedRAGState):
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
        question = state["question"]
        logger.info(f"AdvancedRAGGraph: Node 'retrieve' using index: {self.rag_chain.pinecone_manager.index_name}")
        context = self.rag_chain.retrieve(question)
        return {"context": context}
    
    def _generate_with_context_node(self, state: AdvancedRAGState):
        question = state["question"]
        context = state["context"]
        answer = self.rag_chain.generate(question, context)
        confidence = 0.9 if len(context) >= 2 else 0.6
        return {"answer": answer, "confidence": confidence}
    
    def _generate_without_context_node(self, state: AdvancedRAGState):
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
        if state["needs_retrieval"]:
            return "retrieve"
        return "generate_direct"
    
    def _build_graph(self):
        workflow = StateGraph(AdvancedRAGState)
        workflow.add_node("classify", self._classify_query_node)
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("generate_with_context", self._generate_with_context_node)
        workflow.add_node("generate_direct", self._generate_without_context_node)
        
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
        initial_state = {
            "question": question,
            "context": [],
            "answer": "",
            "needs_retrieval": False,
            "confidence": 0.0
        }
        return self.graph.invoke(initial_state)

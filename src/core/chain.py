import logging
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from .vector_store import PineconeManager

logger = logging.getLogger(__name__)

class RAGChain:
    """
    Implements the RAG chain for question answering.
    """
    
    def __init__(self, pinecone_manager: PineconeManager, model_name: str = "gpt-4o-mini"):
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
    
    def format_context(self, documents: List[Document]) -> str:
        """
        Format retrieved documents into a single context string.
        """
        return "\n\n---\n\n".join([doc.page_content for doc in documents])
    
    def retrieve(self, question: str, k: int = 3) -> List[Document]:
        """
        Retrieve relevant documents for a question.
        """
        logger.info(f"RAGChain: Retrieving context from index: {self.pinecone_manager.index_name}")
        return self.pinecone_manager.similarity_search(question, k=k)
    
    def _get_chain(self):
        return self.prompt | self.llm | StrOutputParser()

    def generate(self, question: str, context: List[Document]) -> str:
        """
        Generate an answer using the LLM.
        """
        formatted_context = self.format_context(context)
        chain = self._get_chain()
        
        return chain.invoke({
            "context": formatted_context,
            "question": question
        })
    
    def query(self, question: str, k: int = 3) -> Dict[str, Any]:
        """
        Execute the full RAG pipeline for a question.
        """
        context = self.retrieve(question, k=k)
        answer = self.generate(question, context)
        
        return {
            "question": question,
            "context": context,
            "answer": answer
        }

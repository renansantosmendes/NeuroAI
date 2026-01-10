import os
import time
from typing import List, Tuple, Optional
from uuid import uuid4

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document

class PineconeManager:
    """
    Manages the creation and querying of the Pinecone vector store.
    """
    
    def __init__(self, pinecone_api_key: str, openai_api_key: str, index_name: str = "rag-langchain-example"):
        self.pinecone_api_key = pinecone_api_key
        self.openai_api_key = openai_api_key
        self.index_name = index_name
        
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.vector_store = None
    
    def create_index(self, dimension: int = 1536, metric: str = "cosine", cloud: str = "aws", region: str = "us-east-1"):
        """
        Create a new Pinecone index if it doesn't exist.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
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
                time.sleep(1)
        
        return self.pc.Index(self.index_name)
    
    def create_from_documents(self, documents: List[Document], namespace: str = "default"):
        """
        Create a vector store from a list of documents.
        """
        self.create_index()
        
        uuids = [str(uuid4()) for _ in range(len(documents))]
        
        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings,
            namespace=namespace
        )
        
        self.vector_store.add_documents(documents=documents, ids=uuids)
        return self.vector_store
    
    def connect_to_existing(self, namespace: str = "default"):
        """
        Connect to an existing Pinecone index.
        """
        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings,
            namespace=namespace
        )
        
        return self.vector_store
    
    def similarity_search(self, query: str, k: int = 3, namespace: str = "default") -> List[Document]:
        """
        Perform similarity search on the vector store.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        return self.vector_store.similarity_search(query, k=k, namespace=namespace)
    
    def similarity_search_with_score(self, query: str, k: int = 3, namespace: str = "default") -> List[Tuple[Document, float]]:
        """
        Perform similarity search and return documents with scores.
        """
        if self.vector_store is None:
            raise ValueError("Vector store not initialized.")
        
        return self.vector_store.similarity_search_with_score(query, k=k, namespace=namespace)
    
    def delete_index(self):
        """
        Delete the Pinecone index.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        if self.index_name in existing_indexes:
            self.pc.delete_index(self.index_name)
    
    def clear_namespace(self, namespace: str = "default"):
        """
        Clear all vectors in a specific namespace.
        """
        index = self.pc.Index(self.index_name)
        index.delete(delete_all=True, namespace=namespace)

    def get_index_stats(self):
        """
        Get statistics about the Pinecone index.
        """
        index = self.pc.Index(self.index_name)
        return index.describe_index_stats()

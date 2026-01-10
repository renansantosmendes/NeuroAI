from typing import List
from langchain_community.document_loaders import TextLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def load_documents(source: str, source_type: str = "text") -> List[Document]:
    """
    Load documents from a file or web URL.
    """
    if source_type == "web":
        loader = WebBaseLoader(source)
    else:
        loader = TextLoader(source)
    
    return loader.load()

def split_documents(documents: List[Document], chunk_size: int = 500, chunk_overlap: int = 50) -> List[Document]:
    """
    Split documents into smaller chunks for better retrieval.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    
    return text_splitter.split_documents(documents)

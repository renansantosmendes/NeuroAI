"""
PDF Processor for Pinecone Indexing.

This module reads PDF files, extracts text, splits into chunks,
generates embeddings and stores them in a Pinecone index.
"""

import os
import time
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

from pinecone import Pinecone, ServerlessSpec


# =============================================================================
# Configuration
# =============================================================================

DEFAULT_INDEX_NAME = "pdf-knowledge-base"
DEFAULT_NAMESPACE = "default"
DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_PINECONE_API_KEY = "pinecone_api_key"
DEFAULT_OPENAI_API_KEY = "openai_api_key"


# =============================================================================
# PDF Loader
# =============================================================================

class PDFLoader:
    """
    Handles loading PDF files from files or directories.
    
    This class provides methods to load single PDFs or
    batch load all PDFs from a directory.
    """
    
    def __init__(self):
        """
        Initialize the PDF loader.
        """
        pass
    
    def load_single_pdf(
        self, 
        file_path: str
    ):
        """
        Load a single PDF file.
        
        Args:
            file_path: Path to the PDF file.
        
        Returns:
            List of Document objects, one per page.
        
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the file is not a PDF.
        
        Example:
            >>> loader = PDFLoader()
            >>> docs = loader.load_single_pdf("document.pdf")
        """
        path = Path(file_path)
        
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        
        if path.suffix.lower() != ".pdf":
            raise ValueError(f"File is not a PDF: {file_path}")
        
        loader = PyPDFLoader(str(path))
        documents = loader.load()
        
        # Add source metadata
        for doc in documents:
            doc.metadata["source_file"] = path.name
            doc.metadata["file_path"] = str(path.absolute())
        
        print(f"Loaded {len(documents)} pages from {path.name}")
        return documents
    
    def load_directory(
        self, 
        directory_path: str, 
        recursive: bool = True
    ):
        """
        Load all PDF files from a directory.
        
        Args:
            directory_path: Path to the directory containing PDFs.
            recursive: If True, search subdirectories as well.
        
        Returns:
            List of Document objects from all PDFs.
        
        Raises:
            FileNotFoundError: If the directory doesn't exist.
        
        Example:
            >>> loader = PDFLoader()
            >>> docs = loader.load_directory("./documents/")
        """
        path = Path(directory_path)
        
        if not path.exists():
            raise FileNotFoundError(f"Directory not found: {directory_path}")
        
        if not path.is_dir():
            raise ValueError(f"Path is not a directory: {directory_path}")
        
        # Find all PDF files
        pattern = "**/*.pdf" if recursive else "*.pdf"
        pdf_files = list(path.glob(pattern))
        
        if not pdf_files:
            print(f"No PDF files found in {directory_path}")
            return []
        
        print(f"Found {len(pdf_files)} PDF files")
        
        all_documents = []
        for pdf_file in pdf_files:
            try:
                docs = self.load_single_pdf(str(pdf_file))
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {pdf_file.name}: {e}")
        
        print(f"Total pages loaded: {len(all_documents)}")
        return all_documents
    
    def load_multiple_pdfs(
        self, 
        file_paths: List[str]
    ):
        """
        Load multiple specific PDF files.
        
        Args:
            file_paths: List of paths to PDF files.
        
        Returns:
            List of Document objects from all PDFs.
        
        Example:
            >>> loader = PDFLoader()
            >>> docs = loader.load_multiple_pdfs(["doc1.pdf", "doc2.pdf"])
        """
        all_documents = []
        
        for file_path in file_paths:
            try:
                docs = self.load_single_pdf(file_path)
                all_documents.extend(docs)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        
        return all_documents


# =============================================================================
# Text Processor
# =============================================================================

class TextProcessor:
    """
    Handles text splitting and preprocessing.
    
    This class splits documents into smaller chunks
    suitable for embedding and retrieval.
    """
    
    def __init__(
        self, 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    ):
        """
        Initialize the text processor.
        
        Args:
            chunk_size: Maximum size of each chunk in characters.
            chunk_overlap: Number of overlapping characters between chunks.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
    
    def split_documents(
        self, 
        documents: List[Document], 
        metadata: dict = None
    ):
        """
        Split documents into smaller chunks.
        
        Args:
            documents: List of Document objects to split.
            metadata: Optional dictionary with custom metadata to add to all chunks.
        
        Returns:
            List of chunked Document objects.
        
        Example:
            >>> processor = TextProcessor(chunk_size=500)
            >>> chunks = processor.split_documents(documents, metadata={"user": "john"})
        """
        chunks = self.text_splitter.split_documents(documents)
        
        # Add chunk index and custom metadata
        for i, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = i
            
            # Add custom metadata if provided
            if metadata:
                chunk.metadata.update(metadata)
        
        print(f"Split {len(documents)} documents into {len(chunks)} chunks")
        return chunks
    
    def clean_text(self, text: str):
        """
        Clean and normalize text content.
        
        Args:
            text: Raw text to clean.
        
        Returns:
            Cleaned text string.
        """
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove common PDF artifacts
        text = text.replace("\x00", "")
        
        return text
    
    def process_documents(
        self, 
        documents: List[Document], 
        metadata: dict = None
    ):
        """
        Clean and split documents with custom metadata.
        
        Args:
            documents: List of Document objects to process.
            metadata: Optional dictionary with custom metadata to add to all chunks.
                      Example: {
                          "application": "medical_app",
                          "user": "dr_smith",
                          "patient": "patient_123",
                          "storage_date": "2025-01-10"
                      }
        
        Returns:
            List of processed and chunked Document objects with metadata.
        
        Example:
            >>> processor = TextProcessor()
            >>> metadata = {
            ...     "application": "my_app",
            ...     "user": "user_123",
            ...     "patient": "patient_456",
            ...     "storage_date": "2025-01-10"
            ... }
            >>> processed = processor.process_documents(raw_docs, metadata=metadata)
        """
        # Clean text in each document
        for doc in documents:
            doc.page_content = self.clean_text(doc.page_content)
        
        # Filter out empty documents
        documents = [doc for doc in documents if doc.page_content.strip()]
        
        # Split into chunks with metadata
        return self.split_documents(documents, metadata=metadata)


# =============================================================================
# Pinecone Indexer
# =============================================================================

class PineconeIndexer:
    """
    Handles indexing documents to Pinecone.
    
    This class manages the Pinecone connection, index creation,
    and document upsert operations.
    """
    
    def __init__(
        self, 
        index_name: str = DEFAULT_INDEX_NAME, 
        pinecone_api_key: str = None, 
        openai_api_key: str = None
    ):
        """
        Initialize the Pinecone indexer.
        
        Args:
            index_name: Name for the Pinecone index.
            pinecone_api_key: Pinecone API key. If None, uses env variable.
            openai_api_key: OpenAI API key. If None, uses env variable.
        """
        self.index_name = index_name
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not set")
        
        if not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")
        
        self.pc = Pinecone(api_key=self.pinecone_api_key)
        self.embeddings = OpenAIEmbeddings(api_key=self.openai_api_key)
        self.vector_store = None
    
    def create_index(
        self, 
        dimension: int = 1536, 
        metric: str = "cosine", 
        cloud: str = "aws", 
        region: str = "us-east-1"
    ):
        """
        Create a new Pinecone index if it doesn't exist.
        
        Args:
            dimension: Vector dimension (1536 for OpenAI ada-002).
            metric: Distance metric - 'cosine', 'euclidean', or 'dotproduct'.
            cloud: Cloud provider - 'aws', 'gcp', or 'azure'.
            region: Cloud region for the index.
        
        Returns:
            The Pinecone index object.
        
        Example:
            >>> indexer = PineconeIndexer("my-index")
            >>> indexer.create_index()
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name not in existing_indexes:
            print(f"Creating new index: {self.index_name}")
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
                time.sleep(2)
            
            print("Index created successfully!")
        else:
            print(f"Using existing index: {self.index_name}")
        
        return self.pc.Index(self.index_name)
    
    def index_documents(
        self, 
        documents: List[Document], 
        namespace: str = DEFAULT_NAMESPACE, 
        batch_size: int = 100
    ):
        """
        Index documents to Pinecone.
        
        Args:
            documents: List of Document objects to index.
            namespace: Pinecone namespace for organizing vectors.
            batch_size: Number of documents to upsert per batch.
        
        Returns:
            The PineconeVectorStore instance.
        
        Example:
            >>> indexer = PineconeIndexer("my-index")
            >>> indexer.create_index()
            >>> indexer.index_documents(chunks)
        """
        # Ensure index exists
        self.create_index()
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index=self.pc.Index(self.index_name),
            embedding=self.embeddings,
            namespace=namespace
        )
        
        # Generate unique IDs
        ids = [str(uuid4()) for _ in range(len(documents))]
        
        # Index in batches
        total_batches = (len(documents) + batch_size - 1) // batch_size
        
        for i in range(0, len(documents), batch_size):
            batch_docs = documents[i:i + batch_size]
            batch_ids = ids[i:i + batch_size]
            batch_num = (i // batch_size) + 1
            
            print(f"Indexing batch {batch_num}/{total_batches} ({len(batch_docs)} documents)...")
            self.vector_store.add_documents(documents=batch_docs, ids=batch_ids)
        
        print(f"Successfully indexed {len(documents)} documents")
        return self.vector_store
    
    def get_index_stats(self):
        """
        Get statistics about the Pinecone index.
        
        Returns:
            Dictionary with index statistics.
        """
        index = self.pc.Index(self.index_name)
        return index.describe_index_stats()
    
    def delete_namespace(
        self, 
        namespace: str = DEFAULT_NAMESPACE
    ):
        """
        Delete all vectors in a namespace.
        
        Args:
            namespace: The namespace to clear.
        """
        index = self.pc.Index(self.index_name)
        index.delete(delete_all=True, namespace=namespace)
        print(f"Deleted all vectors in namespace: {namespace}")
    
    def delete_index(self):
        """
        Delete the entire Pinecone index.
        
        Warning: This permanently deletes all data.
        """
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]
        
        if self.index_name in existing_indexes:
            self.pc.delete_index(self.index_name)
            print(f"Deleted index: {self.index_name}")
        else:
            print(f"Index does not exist: {self.index_name}")


# =============================================================================
# PDF to Pinecone Pipeline
# =============================================================================

class PDFToPineconePipeline:
    """
    Complete pipeline for processing PDFs and indexing to Pinecone.
    
    This class orchestrates the entire workflow from loading PDFs
    to storing embeddings in Pinecone.
    """
    
    def __init__(
        self, 
        index_name: str = DEFAULT_INDEX_NAME, 
        chunk_size: int = DEFAULT_CHUNK_SIZE, 
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP, 
        pinecone_api_key: str = DEFAULT_PINECONE_API_KEY, 
        openai_api_key: str = DEFAULT_OPENAI_API_KEY
    ):
        """
        Initialize the pipeline.
        
        Args:
            index_name: Name for the Pinecone index.
            chunk_size: Size of text chunks.
            chunk_overlap: Overlap between chunks.
            pinecone_api_key: Pinecone API key.
            openai_api_key: OpenAI API key.
        """
        self.pdf_loader = PDFLoader()
        self.text_processor = TextProcessor(chunk_size, chunk_overlap)
        self.indexer = PineconeIndexer(
            index_name=index_name,
            pinecone_api_key=pinecone_api_key,
            openai_api_key=openai_api_key)
    
    def process_single_pdf(
        self, 
        file_path: str, 
        namespace: str = DEFAULT_NAMESPACE, 
        metadata: dict = None
    ):
        """
        Process a single PDF and index to Pinecone.
        
        Args:
            file_path: Path to the PDF file.
            namespace: Pinecone namespace.
            metadata: Optional custom metadata to add to all chunks.
                      Example: {
                          "application": "medical_app",
                          "user": "dr_smith",
                          "patient": "patient_123",
                          "storage_date": "2025-01-10"
                      }
        
        Returns:
            Dictionary with processing statistics.
        
        Example:
            >>> pipeline = PDFToPineconePipeline("my-index")
            >>> metadata = {
            ...     "application": "my_app",
            ...     "user": "user_123",
            ...     "patient": "patient_456",
            ...     "storage_date": "2025-01-10"
            ... }
            >>> stats = pipeline.process_single_pdf("document.pdf", metadata=metadata)
        """
        print(f"\n{'='*60}")
        print(f"Processing: {file_path}")
        print(f"{'='*60}")
        
        # Load PDF
        print("\n1. Loading PDF...")
        documents = self.pdf_loader.load_single_pdf(file_path)
        
        # Process text with metadata
        print("\n2. Processing text...")
        if metadata:
            print(f"   Adding metadata: {metadata}")
        chunks = self.text_processor.process_documents(documents, metadata=metadata)
        
        # Index to Pinecone
        print("\n3. Indexing to Pinecone...")
        self.indexer.index_documents(chunks, namespace)
        
        # Get stats
        stats = self.indexer.get_index_stats()
        
        result = {
            "file": file_path,
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
            "metadata_added": metadata,
            "index_stats": stats
        }
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Pages: {result['pages_loaded']}")
        print(f"Chunks: {result['chunks_created']}")
        print(f"{'='*60}\n")
        
        return result
    
    def process_directory(
        self, 
        directory_path: str, 
        namespace: str = DEFAULT_NAMESPACE, 
        recursive: bool = True, 
        metadata: dict = None
    ):
        """
        Process all PDFs in a directory and index to Pinecone.
        
        Args:
            directory_path: Path to directory containing PDFs.
            namespace: Pinecone namespace.
            recursive: Search subdirectories.
            metadata: Optional custom metadata to add to all chunks.
                      Example: {
                          "application": "medical_app",
                          "user": "dr_smith",
                          "patient": "patient_123",
                          "storage_date": "2025-01-10"
                      }
        
        Returns:
            Dictionary with processing statistics.
        
        Example:
            >>> pipeline = PDFToPineconePipeline("my-index")
            >>> metadata = {"application": "my_app", "user": "admin"}
            >>> stats = pipeline.process_directory("./documents/", metadata=metadata)
        """
        print(f"\n{'='*60}")
        print(f"Processing directory: {directory_path}")
        print(f"{'='*60}")
        
        # Load all PDFs
        print("\n1. Loading PDFs...")
        documents = self.pdf_loader.load_directory(directory_path, recursive)
        
        if not documents:
            print("No documents to process")
            return {"error": "No PDF files found"}
        
        # Process text with metadata
        print("\n2. Processing text...")
        if metadata:
            print(f"   Adding metadata: {metadata}")
        chunks = self.text_processor.process_documents(documents, metadata=metadata)
        
        # Index to Pinecone
        print("\n3. Indexing to Pinecone...")
        self.indexer.index_documents(chunks, namespace)
        
        # Get stats
        stats = self.indexer.get_index_stats()
        
        result = {
            "directory": directory_path,
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
            "metadata_added": metadata,
            "index_stats": stats
        }
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Pages: {result['pages_loaded']}")
        print(f"Chunks: {result['chunks_created']}")
        print(f"{'='*60}\n")
        
        return result
    
    def process_multiple_pdfs(
        self, 
        file_paths: List[str], 
        namespace: str = DEFAULT_NAMESPACE, 
        metadata: dict = None
    ):
        """
        Process multiple specific PDFs and index to Pinecone.
        
        Args:
            file_paths: List of PDF file paths.
            namespace: Pinecone namespace.
            metadata: Optional custom metadata to add to all chunks.
                      Example: {
                          "application": "medical_app",
                          "user": "dr_smith",
                          "patient": "patient_123",
                          "storage_date": "2025-01-10"
                      }
        
        Returns:
            Dictionary with processing statistics.
        
        Example:
            >>> pipeline = PDFToPineconePipeline("my-index")
            >>> metadata = {
            ...     "application": "clinic_app",
            ...     "user": "nurse_jane",
            ...     "patient": "P12345",
            ...     "storage_date": "2025-01-10"
            ... }
            >>> stats = pipeline.process_multiple_pdfs(["doc1.pdf", "doc2.pdf"], metadata=metadata)
        """
        print(f"\n{'='*60}")
        print(f"Processing {len(file_paths)} PDF files")
        print(f"{'='*60}")
        
        # Load PDFs
        print("\n1. Loading PDFs...")
        documents = self.pdf_loader.load_multiple_pdfs(file_paths)
        
        if not documents:
            print("No documents to process")
            return {"error": "No documents loaded"}
        
        # Process text with metadata
        print("\n2. Processing text...")
        if metadata:
            print(f"   Adding metadata: {metadata}")
        chunks = self.text_processor.process_documents(documents, metadata=metadata)
        
        # Index to Pinecone
        print("\n3. Indexing to Pinecone...")
        self.indexer.index_documents(chunks, namespace)
        
        # Get stats
        stats = self.indexer.get_index_stats()
        
        result = {
            "files_processed": len(file_paths),
            "pages_loaded": len(documents),
            "chunks_created": len(chunks),
            "metadata_added": metadata,
            "index_stats": stats
        }
        
        print(f"\n{'='*60}")
        print("Processing complete!")
        print(f"Files: {result['files_processed']}")
        print(f"Pages: {result['pages_loaded']}")
        print(f"Chunks: {result['chunks_created']}")
        print(f"{'='*60}\n")
        
        return result


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """
    Main function with example usage.
    """
    import argparse
    from datetime import datetime
    
    parser = argparse.ArgumentParser(description="Process PDFs and index to Pinecone")
    parser.add_argument("--pdf", type=str, help="Path to a single PDF file")
    parser.add_argument("--directory", type=str, help="Path to directory containing PDFs")
    parser.add_argument("--index-name", type=str, default=DEFAULT_INDEX_NAME, help="Pinecone index name")
    parser.add_argument("--namespace", type=str, default=DEFAULT_NAMESPACE, help="Pinecone namespace")
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE, help="Chunk size for text splitting")
    parser.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP, help="Chunk overlap")
    parser.add_argument("--delete-namespace", action="store_true", help="Delete namespace before indexing")
    
    # Metadata arguments
    parser.add_argument("--application", type=str, help="Application name metadata")
    parser.add_argument("--user", type=str, help="User identifier metadata")
    parser.add_argument("--patient", type=str, help="Patient identifier metadata")
    parser.add_argument("--storage-date", type=str, help="Storage date metadata (YYYY-MM-DD), defaults to today")
    
    args = parser.parse_args()
    
    # Check environment variables
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return
    
    if not os.getenv("PINECONE_API_KEY"):
        print("Error: PINECONE_API_KEY environment variable not set")
        return
    
    # Build metadata dictionary
    metadata = {}
    if args.application:
        metadata["application"] = args.application
    if args.user:
        metadata["user"] = args.user
    if args.patient:
        metadata["patient"] = args.patient
    if args.storage_date:
        metadata["storage_date"] = args.storage_date
    elif args.pdf or args.directory:
        # Default to today's date if processing files
        metadata["storage_date"] = datetime.now().strftime("%Y-%m-%d")
    
    # Use None if no metadata provided
    metadata = metadata if metadata else None
    
    # Initialize pipeline
    pipeline = PDFToPineconePipeline(
        index_name=args.index_name,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    # Delete namespace if requested
    if args.delete_namespace:
        print(f"Deleting namespace: {args.namespace}")
        pipeline.indexer.create_index()
        pipeline.indexer.delete_namespace(args.namespace)
    
    # Process based on input
    if args.pdf:
        result = pipeline.process_single_pdf(args.pdf, args.namespace, metadata=metadata)
    elif args.directory:
        result = pipeline.process_directory(args.directory, args.namespace, metadata=metadata)
    else:
        # Demo mode with sample usage
        print("=" * 60)
        print("PDF to Pinecone Indexer")
        print("=" * 60)
        print("\nUsage examples:")
        print("\n  Process single PDF:")
        print("    python pdf_processor.py --pdf document.pdf")
        print("\n  Process directory:")
        print("    python pdf_processor.py --directory ./documents/")
        print("\n  With metadata:")
        print("    python pdf_processor.py --pdf doc.pdf \\")
        print("      --application medical_app \\")
        print("      --user dr_smith \\")
        print("      --patient patient_123 \\")
        print("      --storage-date 2025-01-10")
        print("\n  With custom settings:")
        print("    python pdf_processor.py --pdf doc.pdf --index-name my-index --chunk-size 500")
        print("\n  Clear namespace before indexing:")
        print("    python pdf_processor.py --directory ./docs/ --delete-namespace")
        print("\nMetadata options:")
        print("  --application    Application name")
        print("  --user           User identifier")
        print("  --patient        Patient identifier")
        print("  --storage-date   Storage date (YYYY-MM-DD)")
        print("\nRequired environment variables:")
        print("  - OPENAI_API_KEY")
        print("  - PINECONE_API_KEY")
        print("=" * 60)
        
        # Example programmatic usage
        print("\n\nProgrammatic usage example:")
        print("-" * 40)
        print("""
from pdf_processor import PDFToPineconePipeline

# Initialize pipeline
pipeline = PDFToPineconePipeline(
    index_name="my-knowledge-base",
    chunk_size=1000,
    chunk_overlap=200
)

# Define metadata
metadata = {
    "application": "medical_app",
    "user": "dr_smith",
    "patient": "patient_123",
    "storage_date": "2025-01-10"
}

# Process single PDF with metadata
result = pipeline.process_single_pdf(
    "document.pdf",
    metadata=metadata
)

# Process directory with metadata
result = pipeline.process_directory(
    "./documents/",
    metadata=metadata
)

# Process multiple specific files with metadata
result = pipeline.process_multiple_pdfs(
    ["doc1.pdf", "doc2.pdf"],
    metadata=metadata
)
""")


if __name__ == "__main__":
    main()
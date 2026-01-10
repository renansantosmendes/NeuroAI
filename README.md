# NeuroAI RAG Agent API

This project implements a robust Retrieval-Augmented Generation (RAG) system using LangChain, LangGraph, and Pinecone, served through a FastAPI application.

## Project Structure

- `src/`
  - `api/`: FastAPI application, endpoints, and schemas.
  - `core/`: Core RAG logic, including chains, graphs, and vector store management.
  - `config.py`: Environment configuration and validation.
  - `main.py`: Application entry point.
- `tests/`: Comprehensive unit test suite.

## Features

- **Standard RAG**: Basic retrieval and generation pipeline.
- **Stateful RAG (LangGraph)**: Workflow-based RAG for better state management.
- **Advanced RAG**: Intelligent routing to determine if document retrieval is necessary.
- **Vector Store**: Seamless integration with Pinecone.
- **REST API**: Well-documented endpoints for all RAG functionalities.

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure environment variables in a `.env` file:
   ```env
   OPENAI_API_KEY=your_openai_key
   PINECONE_API_KEY=your_pinecone_key
   PINECONE_INDEX_NAME=rag-index
   ```

## Running the API

Start the FastAPI server:
```bash
python -m src.main
```

The API will be available at `http://localhost:8000`.
You can access the interactive documentation (Swagger UI) at `http://localhost:8000/docs`.

## API Endpoints

### 1. Basic Query
- **URL**: `/query`
- **Method**: `POST`
- **Body**: `{"question": "string", "k": int}`
- **Description**: Standard RAG pipeline using a direct chain.

### 2. Graph-based Query
- **URL**: `/query/graph`
- **Method**: `POST`
- **Body**: `{"question": "string"}`
- **Description**: Uses LangGraph to manage the RAG workflow state.

### 3. Advanced Query
- **URL**: `/query/advanced`
- **Method**: `POST`
- **Body**: `{"question": "string"}`
- **Description**: Classifies the query and routes it either to document retrieval or direct answer.

## Testing

Run the test suite with coverage:
```bash
python -m pytest --cov=src tests/
```

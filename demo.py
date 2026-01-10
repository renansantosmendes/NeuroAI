import chainlit as cl
from src.config import Config
from src.core.vector_store import PineconeManager
from src.core.chain import RAGChain
from src.core.graph import AdvancedRAGGraph

# Initialize components once logic
def get_rag_system():
    Config.validate()
    pm = PineconeManager(
        pinecone_api_key=Config.PINECONE_API_KEY,
        openai_api_key=Config.OPENAI_API_KEY,
        index_name=Config.INDEX_NAME
    )
    # Connect to index (assuming it exists)
    pm.connect_to_existing()
    rag_chain = RAGChain(pm)
    advanced_rag = AdvancedRAGGraph(rag_chain)
    return advanced_rag

@cl.on_chat_start
async def on_chat_start():
    """
    Initialize the RAG system and store it in the user session.
    """
    try:
        # Show a loading message
        msg = cl.Message(content="Initializing RAG Agent...")
        await msg.send()
        
        # Load the system
        rag_system = get_rag_system()
        
        # Store in session
        cl.user_session.set("rag_system", rag_system)
        
        msg.content = "RAG Agent ready! How can I help you today?"
        await msg.update()
    except Exception as e:
        await cl.Message(content=f"Error initializing system: {str(e)}").send()

@cl.on_message
async def main(message: cl.Message):
    """
    Process user messages using the RAG system.
    """
    # Get the system from session
    rag_system = cl.user_session.get("rag_system")
    
    if not rag_system:
        await cl.Message(content="System not initialized. Please refresh.").send()
        return

    # Create an empty message for streaming or intermediate status
    msg = cl.Message(content="")
    await msg.send()

    # Define a step for the RAG process
    async with cl.Step(name="RAG Process") as step:
        # Run the advanced RAG graph
        # Note: Since the graph is synchronous in the implementation, we run it normally.
        # If it were asynchronous, we would await it.
        result = rag_system.query(message.content)
        
        step.input = message.content
        step.output = f"Retrieval needed: {result.get('needs_retrieval', 'N/A')}\nConfidence: {result.get('confidence', 0):.2f}"

    # Send the final answer
    msg.content = result["answer"]
    
    # Optional: Add retrieved documents as elements
    elements = []
    if result.get("context"):
        for i, doc in enumerate(result["context"]):
            elements.append(
                cl.Text(name=f"Source {i+1}", content=doc.page_content, display="side")
            )
    
    msg.elements = elements
    await msg.update()
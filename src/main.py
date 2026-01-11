import uvicorn
import os
import logging
from src.api.app import app
from src.config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def start():
    """
    Start the FastAPI server.
    """
    # Validate config on startup
    try:
        Config.validate()
        print("Configuration validated successfully.")
    except ValueError as e:
        print(f"Configuration error: {e}")
        print("Please ensure your .env file is correctly configured.")
        # We don't exit here to allow the app to potentially use defaults or show errors via API
    
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    
    print(f"Starting NeuroAI RAG API on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    start()

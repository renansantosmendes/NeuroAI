import uvicorn
import os
from .api.app import app
from .config import Config

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

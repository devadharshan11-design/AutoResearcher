import os
from dotenv import load_dotenv

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_PATH = os.path.join(BASE_DIR, ".env")
load_dotenv(ENV_PATH)

# Ollama model configuration
OLLAMA_MODEL_NAME = os.getenv("OLLAMA_MODEL_NAME", "llama3")

# Embedding model configuration
EMBEDDING_MODEL_NAME = os.getenv(
    "EMBEDDING_MODEL_NAME",
    "sentence-transformers/all-MiniLM-L6-v2"
)

# Vector database directory
VECTOR_DB_DIR = os.getenv(
    "VECTOR_DB_DIR",
    os.path.join("data", "index")
)

"""
Configuration settings for the RAG Chat API
"""
from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()


class Settings(BaseSettings):
    """Application settings"""
    
    # OpenAI Configuration
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o")

    # Ollama (local, free) - set USE_OLLAMA=true when OpenAI quota exceeded or no key
    use_ollama: bool = os.getenv("USE_OLLAMA", "false").lower() == "true"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.2:1b")
    
    # Embedding Configuration
    embedding_model: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    use_openai_embeddings: bool = os.getenv("USE_OPENAI_EMBEDDINGS", "false").lower() == "true"
    
    # Vector Store Configuration
    vector_store_path: str = os.getenv("VECTOR_STORE_PATH", "./vector_store")
    collection_name: str = os.getenv("COLLECTION_NAME", "documents")
    
    # RAG Configuration
    top_k_results: int = int(os.getenv("TOP_K_RESULTS", "5"))
    similarity_threshold: float = float(os.getenv("SIMILARITY_THRESHOLD", "0.7"))

    # Smart ranking (cross-encoder reranker for better relevance)
    use_reranker: bool = os.getenv("USE_RERANKER", "true").lower() == "true"
    reranker_model: str = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")
    reranker_top_k: int = int(os.getenv("RERANKER_TOP_K", "5"))
    
    # Server Configuration
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    
    class Config:
        env_file = ".env"
        case_sensitive = False


settings = Settings()

# Use current model if .env still has a deprecated name
if settings.openai_model in ("gpt-4-turbo-preview", "gpt-4-turbo"):
    settings.openai_model = "gpt-4o"

# If using Ollama with a model name that's often not installed, use 1b variant (common default)
if settings.use_ollama and settings.ollama_model in ("llama3.2:latest", "llama3.2"):
    settings.ollama_model = "llama3.2:1b"

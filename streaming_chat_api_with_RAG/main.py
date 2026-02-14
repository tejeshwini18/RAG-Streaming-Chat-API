"""
Main FastAPI application for Streaming Chat API with RAG
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from api.routes import router
from config import settings

app = FastAPI(
    title="Streaming Chat API with RAG",
    description="Distributed LLM + Streaming Chat API with Retrieval-Augmented Generation",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routes
app.include_router(router, prefix="/api/v1", tags=["chat"])

# Serve UI at root
STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
async def root():
    """Serve the chat UI."""
    index_path = STATIC_DIR / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return {
        "message": "Streaming Chat API with RAG",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/v1/chat",
            "add_documents": "/api/v1/documents/add",
            "search": "/api/v1/documents/search",
            "health": "/api/v1/health"
        }
    }


@app.get("/api")
async def api_info():
    """API info (JSON) for programmatic use."""
    return {
        "message": "Streaming Chat API with RAG",
        "version": "1.0.0",
        "endpoints": {
            "chat": "/api/v1/chat",
            "add_documents": "/api/v1/documents/add",
            "search": "/api/v1/documents/search",
            "health": "/api/v1/health"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )

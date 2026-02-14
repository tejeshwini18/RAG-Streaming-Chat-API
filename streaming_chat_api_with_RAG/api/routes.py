"""
API routes for streaming chat with RAG
"""
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import json
import logging

from rag.embedder import Embedder
from rag.vector_store import VectorStore
from rag.retriever import Retriever
from llm.streaming_llm import StreamingLLM
from config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize components (lazy so missing API key doesn't break startup)
_embedder = None
_vector_store = None
_retriever = None
_llm = None


def _get_components():
    global _embedder, _vector_store, _retriever, _llm
    if _embedder is None:
        _embedder = Embedder()
        _vector_store = VectorStore(_embedder)
        _retriever = Retriever(_vector_store)
        _llm = StreamingLLM()
    return _embedder, _vector_store, _retriever, _llm


class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str


class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    stream: bool = True
    temperature: Optional[float] = None
    top_k: Optional[int] = None
    debug: bool = False
    model: Optional[str] = None  # Override model for this request (Ollama or OpenAI)


class DocumentRequest(BaseModel):
    documents: List[str]
    metadatas: Optional[List[dict]] = None


@router.post("/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint with RAG
    
    Flow:
    1. Extract user query from messages
    2. Embed query
    3. Retrieve relevant documents
    4. Format context
    5. Stream LLM response with context
    """
    if not request.messages:
        raise HTTPException(status_code=400, detail="Messages cannot be empty")

    user_messages = [msg for msg in request.messages if msg.role == "user"]
    if not user_messages:
        raise HTTPException(status_code=400, detail="No user message found")

    if not settings.use_ollama and not (settings.openai_api_key or settings.openai_api_key.strip()):
        raise HTTPException(
            status_code=503,
            detail="OPENAI_API_KEY is not set. Add it to your .env file and restart the server. Or set USE_OLLAMA=true to use a local model (Ollama).",
        )

    try:
        embedder, vector_store, retriever, llm = _get_components()
    except Exception as e:
        logger.exception("Failed to initialize RAG/LLM components")
        raise HTTPException(status_code=503, detail=f"Server not ready: {str(e)}")

    query = user_messages[-1].content
    top_k = request.top_k if request.top_k is not None else settings.top_k_results
    temperature = request.temperature if request.temperature is not None else 0.7
    model_override = (request.model or "").strip() or None
    try:
        retrieved_docs = retriever.retrieve(query, top_k=top_k)
        context = retriever.format_context(retrieved_docs)
    except Exception as e:
        logger.exception("Retrieval failed")
        raise HTTPException(status_code=500, detail=f"Retrieval failed: {str(e)}")

    messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]

    def _sources_payload():
        if not retrieved_docs:
            return None
        items = []
        for d in retrieved_docs:
            meta = d.get("metadata") or {}
            name = meta.get("source") or meta.get("filename") or d.get("id") or "doc"
            page = meta.get("page") or meta.get("chunk")
            items.append({
                "id": d.get("id"),
                "name": str(name),
                "page": str(page) if page is not None else None,
                "similarity": round(d.get("similarity", 0), 4),
                "preview": (d.get("content") or "")[:150] + ("…" if len(d.get("content") or "") > 150 else ""),
            })
        return {"count": len(retrieved_docs), "items": items}

    if request.stream:
        async def generate():
            try:
                sources = _sources_payload()
                if sources:
                    yield f"data: {json.dumps({'sources': sources})}\n\n"
                if request.debug and context:
                    yield f"data: {json.dumps({'debug_context': context[:2000] + ('…' if len(context) > 2000 else '')})}\n\n"
                async for chunk in llm.stream_response(messages, context, temperature=temperature, model=model_override):
                    yield f"data: {json.dumps({'content': chunk})}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as e:
                logger.exception("Streaming failed")
                err_msg = str(e)
                if "429" in err_msg or "quota" in err_msg.lower() or "insufficient_quota" in err_msg.lower():
                    err_msg = (
                        "OpenAI quota exceeded. Check billing: https://platform.openai.com/account/billing — "
                        "Or use a free local model: set USE_OLLAMA=true in .env and run Ollama (https://ollama.com)."
                    )
                elif "404" in err_msg and ("not found" in err_msg.lower() or "model" in err_msg.lower()):
                    err_msg = (
                        "Ollama model not found. Run 'ollama list' to see installed models, then set OLLAMA_MODEL in .env (e.g. llama3.2:latest). "
                        "To download a model, run: ollama pull llama3.2"
                    )
                yield f"data: {json.dumps({'error': err_msg})}\n\n"
                yield "data: [DONE]\n\n"

        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
            },
        )
    else:
        try:
            response = llm.generate_response(messages, context, temperature=temperature, model=model_override)
            out = {"response": response, "context_used": len(retrieved_docs) > 0, "sources": _sources_payload()}
            if request.debug and context:
                out["debug_context"] = context[:3000] + ("…" if len(context) > 3000 else "")
            return out
        except Exception as e:
            logger.exception("LLM request failed")
            raise HTTPException(status_code=500, detail=str(e))


@router.post("/documents/add")
async def add_documents(request: DocumentRequest):
    """Add documents to the vector store"""
    try:
        _, vector_store, _, _ = _get_components()
        vector_store.add_documents(
            documents=request.documents,
            metadatas=request.metadatas
        )
        return {
            "status": "success",
            "documents_added": len(request.documents)
        }
    except Exception as e:
        logger.exception("Add documents failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/documents/search")
async def search_documents(query: str, top_k: int = 5):
    """Search documents without generating a response"""
    _, _, retriever, _ = _get_components()
    retrieved_docs = retriever.retrieve(query, top_k=top_k)
    return {
        "query": query,
        "results": retrieved_docs,
        "count": len(retrieved_docs)
    }


@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "vector_store": "connected",
        "llm": "ready"
    }


@router.get("/documents/list")
async def list_documents():
    """List all indexed documents with metadata."""
    try:
        embedder, vector_store, _, _ = _get_components()
        docs = vector_store.list_documents()
        dim = getattr(embedder, "get_embedding_dimension", lambda: None)()
        return {
            "documents": docs,
            "total": len(docs),
            "embedding_dimension": dim,
        }
    except Exception as e:
        logger.exception("List documents failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/documents/{doc_id}")
async def delete_document(doc_id: str):
    """Delete one document by id."""
    try:
        _, vector_store, _, _ = _get_components()
        ok = vector_store.delete_document(doc_id)
        if not ok:
            raise HTTPException(status_code=404, detail="Document not found")
        return {"status": "deleted", "id": doc_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Delete document failed")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/system/stats")
async def system_stats():
    """System dashboard: model, inference stats (mock/real)."""
    try:
        embedder, vector_store, retriever, llm = _get_components()
        model_name = settings.ollama_model if settings.use_ollama else settings.openai_model
        dim = getattr(embedder, "get_embedding_dimension", lambda: None)()
        doc_count = len(vector_store.get_all_documents())
        retrieval_quality = "Standard"
        if getattr(settings, "use_reranker", False):
            try:
                from rag.pytorch_reranker import PyTorchReranker
                PyTorchReranker()
                retrieval_quality = "Boosted"
            except Exception:
                retrieval_quality = "Boosted (loading…)"
        return {
            "model_loaded": model_name,
            "backend": "ollama" if settings.use_ollama else "openai",
            "embedding_dimension": dim,
            "documents_indexed": doc_count,
            "gpu_active": settings.use_ollama,
            "distributed_mode": "OFF",
            "inference_latency_ms": None,
            "tokens_per_sec": None,
            "retrieval_quality": retrieval_quality,
        }
    except Exception as e:
        logger.exception("System stats failed")
        raise HTTPException(status_code=500, detail=str(e))


# Popular models to show in dropdown (user can select and e.g. run `ollama pull <name>` if not installed)
OLLAMA_SUGGESTED = [
    "llama3.2:1b",
    "llama3.2:3b",
    "llama3.2:latest",
    "llama3.1:8b",
    "llama3.1:latest",
    "mistral",
    "mistral:7b",
    "mixtral:8x7b",
    "codellama",
    "phi3",
    "phi3:mini",
    "gemma:2b",
    "gemma:7b",
    "qwen2:7b",
    "deepseek-coder",
]

OPENAI_SUGGESTED = [
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo",
    "gpt-3.5-turbo",
]


@router.get("/ollama/models")
async def ollama_models():
    """List installed Ollama models."""
    if not settings.use_ollama:
        return {"models": [], "message": "Ollama not enabled."}
    try:
        import httpx
        r = httpx.get(settings.ollama_base_url.replace("/v1", "") + "/api/tags", timeout=5.0)
        if r.status_code != 200:
            return {"models": [], "error": r.text}
        data = r.json()
        models = [m.get("name", "") for m in data.get("models", [])]
        return {"models": models}
    except Exception as e:
        return {"models": [], "error": str(e)}


@router.get("/models")
async def list_models():
    """List installed + suggested models for the dropdown (backend-aware)."""
    backend = "ollama" if settings.use_ollama else "openai"
    default = settings.ollama_model if settings.use_ollama else settings.openai_model
    installed = []
    if settings.use_ollama:
        try:
            import httpx
            r = httpx.get(settings.ollama_base_url.replace("/v1", "") + "/api/tags", timeout=5.0)
            if r.status_code == 200:
                installed = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception:
            pass
        suggested = [m for m in OLLAMA_SUGGESTED if m not in installed]
        return {"backend": backend, "default": default, "installed": installed, "suggested": suggested}
    else:
        installed = [default] if default else []
        return {"backend": backend, "default": default, "installed": installed, "suggested": OPENAI_SUGGESTED}

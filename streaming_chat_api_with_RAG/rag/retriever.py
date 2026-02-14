"""
Retriever for fetching relevant documents based on queries.
Optionally uses a PyTorch cross-encoder reranker for better ranking.
"""
from typing import List, Dict, Optional
from config import settings
from .vector_store import VectorStore


class Retriever:
    """Retrieves relevant documents based on query similarity; optional PyTorch reranker."""

    def __init__(self, vector_store: VectorStore, reranker=None):
        self.vector_store = vector_store
        self._reranker = reranker

    @property
    def reranker(self):
        if self._reranker is None and getattr(settings, "use_reranker", False):
            try:
                from .pytorch_reranker import PyTorchReranker
                self._reranker = PyTorchReranker(
                    model_name=getattr(settings, "reranker_model", None),
                    max_length=512,
                )
            except Exception:
                self._reranker = False  # disabled on failure
        return self._reranker if self._reranker else None

    def retrieve(self, query: str, top_k: int = None) -> List[Dict]:
        """
        Retrieve relevant documents for a query.
        If USE_RERANKER is true, fetches more candidates then reranks with PyTorch.
        """
        if top_k is None:
            top_k = settings.top_k_results

        # Fetch more candidates if we will rerank (so we have enough to rerank)
        fetch_k = top_k
        if getattr(settings, "use_reranker", False) and self.reranker:
            fetch_k = max(top_k * 2, top_k + 5)

        results = self.vector_store.query(query, n_results=fetch_k)

        retrieved_docs = []
        if results.get("documents") and len(results["documents"][0]) > 0:
            documents = results["documents"][0]
            metadatas = results.get("metadatas", [[]])[0] or [{}] * len(documents)
            distances = results.get("distances", [[]])[0] or [1.0] * len(documents)
            ids = results.get("ids", [[]])[0] or [""] * len(documents)
            for doc, metadata, distance, doc_id in zip(documents, metadatas, distances, ids):
                similarity = 1 - distance if distance <= 1 else 1 / (1 + distance)
                if similarity >= settings.similarity_threshold:
                    retrieved_docs.append({
                        "id": doc_id,
                        "content": doc,
                        "metadata": metadata,
                        "similarity": similarity,
                        "distance": distance,
                    })

        if self.reranker and retrieved_docs:
            reranker_top_k = getattr(settings, "reranker_top_k", top_k)
            retrieved_docs = self.reranker.rerank(query, retrieved_docs, top_k=reranker_top_k)

        return retrieved_docs[:top_k]
    
    def format_context(self, retrieved_docs: List[Dict]) -> str:
        """Format retrieved documents into context string"""
        if not retrieved_docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(retrieved_docs, 1):
            content = doc["content"]
            metadata = doc.get("metadata", {})
            similarity = doc.get("similarity", 0)
            
            context_parts.append(
                f"[Document {i} (Similarity: {similarity:.2f})]\n{content}\n"
            )
        
        return "\n".join(context_parts)

"""
Vector store for storing and retrieving document embeddings.
Pure Python + NumPy implementation (no C++ build required).
"""
import json
import os
import time
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np

from config import settings
from .embedder import Embedder


class VectorStore:
    """Manages vector storage using NumPy (persisted to disk)."""

    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self._store_path = Path(settings.vector_store_path)
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._embeddings_path = self._store_path / "embeddings.npy"
        self._meta_path = self._store_path / "meta.json"
        self._embeddings: np.ndarray = np.array([])
        self._documents: List[str] = []
        self._metadatas: List[Dict] = []
        self._ids: List[str] = []
        self._load()

    def _load(self) -> None:
        """Load persisted store from disk."""
        if self._embeddings_path.exists() and self._meta_path.exists():
            try:
                self._embeddings = np.load(self._embeddings_path)
                with open(self._meta_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self._documents = data.get("documents", [])
                self._metadatas = data.get("metadatas", [])
                self._ids = data.get("ids", [])
            except Exception:
                self._embeddings = np.array([])
                self._documents = []
                self._metadatas = []
                self._ids = []

    def _save(self) -> None:
        """Persist store to disk."""
        if len(self._documents) == 0:
            return
        np.save(self._embeddings_path, self._embeddings)
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "documents": self._documents,
                    "metadatas": self._metadatas,
                    "ids": self._ids,
                },
                f,
                ensure_ascii=False,
            )

    def add_documents(
        self,
        documents: List[str],
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None,
    ) -> None:
        """Add documents to the vector store."""
        if not documents:
            return

        embeddings = self.embedder.embed_documents(documents)
        if ids is None:
            base = len(self._ids)
            ids = [f"doc_{base + i}" for i in range(len(documents))]
        if metadatas is None:
            metadatas = [{}] * len(documents)
        for m in metadatas:
            if "indexed_at" not in m:
                m["indexed_at"] = time.time()

        arr = np.array(embeddings, dtype=np.float64)
        if self._embeddings.size == 0:
            self._embeddings = arr
        else:
            self._embeddings = np.vstack([self._embeddings, arr])

        self._documents.extend(documents)
        self._metadatas.extend(metadatas)
        self._ids.extend(ids)
        self._save()

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        filter: Optional[Dict] = None,
    ) -> Dict:
        """Query the vector store. Returns ChromaDB-like structure for compatibility."""
        if self._embeddings.size == 0:
            return {
                "documents": [[]],
                "metadatas": [[]],
                "distances": [[]],
                "ids": [[]],
            }

        query_embedding = np.array(
            self.embedder.embed_query(query_text), dtype=np.float64
        )
        query_embedding = query_embedding.reshape(1, -1)

        # Cosine similarity (ChromaDB often uses distance = 1 - similarity)
        similarities = np.dot(self._embeddings, query_embedding.T).flatten()
        norms = np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(
            query_embedding
        )
        norms = np.maximum(norms, 1e-10)
        similarities = similarities / norms
        # Return as distance for compatibility: distance = 1 - similarity
        distances = 1.0 - similarities

        top_indices = np.argsort(distances)[:n_results]
        if filter is not None:
            # Simple filter: match metadata key-value on first doc
            filtered = []
            for i in top_indices:
                meta = self._metadatas[i]
                if all(meta.get(k) == v for k, v in filter.items()):
                    filtered.append(i)
            top_indices = np.array(filtered[:n_results]) if filtered else top_indices[:0]

        doc_results = [self._documents[i] for i in top_indices]
        meta_results = [self._metadatas[i] for i in top_indices]
        dist_results = [float(distances[i]) for i in top_indices]
        id_results = [self._ids[i] for i in top_indices]

        return {
            "documents": [doc_results],
            "metadatas": [meta_results],
            "distances": [dist_results],
            "ids": [id_results],
        }

    def get_all_documents(self) -> List[str]:
        """Get all documents from the store."""
        return list(self._documents)

    def list_documents(self) -> List[Dict]:
        """List all documents with id, preview, metadata, index."""
        out = []
        for i, (doc_id, doc, meta) in enumerate(zip(self._ids, self._documents, self._metadatas)):
            out.append({
                "id": doc_id,
                "content_preview": (doc or "")[:200] + ("…" if len(doc or "") > 200 else ""),
                "metadata": meta or {},
                "index": i,
            })
        return out

    def delete_document(self, doc_id: str) -> bool:
        """Remove one document by id. Returns True if removed."""
        try:
            idx = self._ids.index(doc_id)
        except ValueError:
            return False
        self._ids.pop(idx)
        self._documents.pop(idx)
        self._metadatas.pop(idx)
        self._embeddings = np.delete(self._embeddings, idx, axis=0)
        self._save()
        return True

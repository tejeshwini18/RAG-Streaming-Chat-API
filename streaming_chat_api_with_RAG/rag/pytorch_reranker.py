"""
PyTorch-based cross-encoder reranker for RAG.
Rescores (query, document) pairs using a small BERT-style model on PyTorch.
"""
from typing import List, Dict, Optional
import logging

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import settings

logger = logging.getLogger(__name__)


class PyTorchReranker:
    """
    Rerank candidate documents using a PyTorch cross-encoder.
    Uses explicit torch tensors and .to(device) for visibility of PyTorch in the stack.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        max_length: int = 512,
    ):
        self.model_name = model_name or getattr(
            settings, "reranker_model", "cross-encoder/ms-marco-MiniLM-L-6-v2"
        )
        self.max_length = max_length
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._tokenizer = None

    @property
    def model(self):
        if self._model is None:
            logger.info("Loading PyTorch reranker: %s on %s", self.model_name, self.device)
            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._model = AutoModelForSequenceClassification.from_pretrained(self.model_name)
            self._model.to(self.device)
            self._model.eval()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            _ = self.model  # trigger load
        return self._tokenizer

    def score_pairs(self, query: str, documents: List[str]) -> torch.Tensor:
        """
        Score (query, doc) pairs with the cross-encoder. Returns a 1D tensor of scores.
        Uses (query_list, passage_list) format expected by MS MARCO-style models.
        """
        if not documents:
            return torch.tensor([], dtype=torch.float32, device=self.device)

        queries = [query] * len(documents)
        encoded = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        # Explicit PyTorch: move inputs to device
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with torch.no_grad():
            outputs = self.model(**encoded)
            # logits shape: (batch_size, 1) -> squeeze to (batch_size,)
            scores = outputs.logits.squeeze(-1)

        return scores

    def rerank(
        self,
        query: str,
        candidates: List[Dict],
        top_k: Optional[int] = None,
    ) -> List[Dict]:
        """
        Rerank candidate docs (list of dicts with 'content'). Returns same dicts
        with updated order and added 'rerank_score' key. Uses PyTorch for scoring.
        """
        if not candidates:
            return []
        top_k = top_k or len(candidates)

        texts = [c.get("content") or "" for c in candidates]
        scores_tensor = self.score_pairs(query, texts)
        # Move to CPU and convert to list for easy use
        scores_list = scores_tensor.cpu().tolist()
        if isinstance(scores_list, float):
            scores_list = [scores_list]

        for c, sc in zip(candidates, scores_list):
            c["rerank_score"] = sc
            # Optionally overwrite similarity with normalized rerank score for display
            c["similarity"] = self._sigmoid(sc)

        # Sort by rerank score descending
        candidates_sorted = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
        return candidates_sorted[:top_k]

    @staticmethod
    def _sigmoid(x: float) -> float:
        """Map logit to 0-1 for display as similarity."""
        import math
        return 1.0 / (1.0 + math.exp(-x))

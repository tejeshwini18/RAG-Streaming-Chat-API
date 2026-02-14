"""
Embedding service for converting text to vectors
"""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config import settings


class Embedder:
    """Handles text embedding using either local or OpenAI models"""
    
    def __init__(self):
        if settings.use_openai_embeddings:
            self.client = OpenAI(api_key=settings.openai_api_key)
            self.model = None
        else:
            self.client = None
            self.model = SentenceTransformer(settings.embedding_model)
    
    def embed_query(self, text: str) -> List[float]:
        """Embed a single query text"""
        if self.client:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=text
            )
            return response.data[0].embedding
        else:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed multiple documents"""
        if self.client:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=texts
            )
            return [item.embedding for item in response.data]
        else:
            embeddings = self.model.encode(texts, convert_to_numpy=True)
            return embeddings.tolist()
    
    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings"""
        if self.client:
            # OpenAI ada-002 has 1536 dimensions
            return 1536
        else:
            # Get dimension from model
            test_embedding = self.model.encode("test", convert_to_numpy=True)
            return len(test_embedding)

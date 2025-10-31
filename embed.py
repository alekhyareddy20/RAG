"""
Custom Embedding Function Module
---------------------------------
This module provides a custom embedding function that connects ChromaDB
to Hugging Face's Inference API for generating text embeddings using
the Qwen embedding model.
"""

from typing import List
import numpy as np
from huggingface_hub import InferenceClient
from chromadb.api.types import Embeddings


class HuggingFaceEmbeddingFunction:
    """
    Custom embedding function for ChromaDB integration with Hugging Face models.
    
    This class handles the conversion of text into vector embeddings using
    Hugging Face's Inference API. It's compatible with ChromaDB's embedding
    function interface.
    
    Attributes:
        inference_client (InferenceClient): Client for making API calls to Hugging Face
    """
    
    def __init__(self, model_id: str):
        """
        Initialize the embedding function with a specific Hugging Face model.
        
        Args:
            model_id (str): Hugging Face model identifier (e.g., "Qwen/Qwen3-Embedding-8B")
        """
        # Initialize the Hugging Face Inference Client once during instantiation
        self.inference_client = InferenceClient(model=model_id)

    def embed_documents(self, text_list: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.
        This method is called by ChromaDB when adding documents to the database.
        
        Args:
            text_list (List[str]): List of text strings to embed
            
        Returns:
            List[List[float]]: List of embedding vectors (one per input text)
        """
        # Call Hugging Face API to generate embeddings
        # Note: Passing texts as positional argument (not keyword) for compatibility
        embedding_vectors = self.inference_client.feature_extraction(text_list)
        
        # Convert numpy arrays to lists for ChromaDB compatibility
        if isinstance(embedding_vectors, np.ndarray):
            return embedding_vectors.tolist()
        elif isinstance(embedding_vectors, List) and embedding_vectors and isinstance(embedding_vectors[0], np.ndarray):
            return [vector.tolist() for vector in embedding_vectors]

        return embedding_vectors

    def embed_query(self, query_text: str) -> List[float]:
        """
        Generate embedding for a single query string.
        This method is called by ChromaDB when performing similarity searches.
        
        Args:
            query_text (str): Single text string to embed
            
        Returns:
            List[float]: Single embedding vector
        """
        # Feature extraction accepts a list, so wrap the single query
        embedding_result = self.inference_client.feature_extraction([query_text])
        
        # Return the first (and only) embedding vector
        return embedding_result[0]
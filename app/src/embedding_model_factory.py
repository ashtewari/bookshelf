# embedding_model_factory.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
import os
import torch

class EmbeddingModelFactory:
    @staticmethod
    def create_instance(model_name, api_base=None, api_key=None, cuda_is_available=False, embed_batch_size=100, max_length=512, timeout=30):
    
        if (model_name == "OpenAIEmbedding"):
            embed_model = OpenAIEmbedding(api_base=api_base, 
                                          api_key=api_key, 
                                          timeout=timeout, 
                                          embed_batch_size=embed_batch_size)
        else:
            embed_model = HuggingFaceEmbedding(
                model_name=model_name,
                embed_batch_size=embed_batch_size,
                max_length=max_length,
                device='cuda' if cuda_is_available else 'cpu'
                )
        return embed_model
      
    @staticmethod
    def get_embedding_model(collection, embedding_model_requested):
        """
        Retrieves the embedding model from the collection metadata.
        If no embedding model is found, returns the requested embedding model.
        """
        embedding_models = set()
        try:
            # Get all items from collection with an empty where clause that matches everything
            docs = collection.get(where={})
            
            # Access metadatas from the docs
            metadatas = docs.get('metadatas', [])
            if metadatas:
                for metadata in metadatas:
                    if metadata and 'embedding_model' in metadata:
                        embedding_models.add(metadata['embedding_model'])
        except Exception:
            # If there's any error accessing the collection, fall back to requested model
            return embedding_model_requested
        
        if embedding_models:
            # Use the first embedding model found to generate query embeddings
            model_name = list(embedding_models)[0]
            embedding_model = EmbeddingModelFactory.create_instance(
                model_name=model_name,
                api_base=os.getenv('OPENAI_API_BASE'),
                api_key=os.getenv('OPENAI_API_KEY'),
                cuda_is_available=torch.cuda.is_available()
            )
        else:
            embedding_model = embedding_model_requested
        
        return embedding_model      
# embedding_model_factory.py
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding

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
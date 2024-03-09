import chromadb 
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class ChromaDb:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path)

    ## get list of collections
    def get_collections(self):
        collections = []

        for i in self.client.list_collections():
            collections.append(i.name)
        
        return collections
    
    ## get documents in specified collection
    def get_collection_data(self, collection_name, dataframe=False):
        data = self.client.get_collection(name=collection_name).get()
        if dataframe:
            return pd.DataFrame(data)
        return data
    
    ## query specified collection
    def query(self, query_str, collection_name, model_name, k=3, dataframe=False):
        collection = self.client.get_collection(collection_name)
        
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        embedding = embed_model.get_text_embedding(query_str)
        res = collection.query(
            query_embeddings=[embedding], n_results=min(k, len(collection.get()))
        )
        out = {}
        for key, value in res.items():
            if value:
                out[key] = value[0]
            else:
                out[key] = value
        if dataframe:
            return pd.DataFrame(out)
        return out
    
    def delete_collection(self, collection_name):
        self.client.delete_collection(name=collection_name)
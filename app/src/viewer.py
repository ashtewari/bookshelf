import os
import chromadb 
import pandas as pd
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

class ChromaDb:
    def __init__(self, path):
        self.client = chromadb.PersistentClient(path)

    ## get list of collections
    def get_collections(self) -> list[dict]:
        collections = []

        for i in self.client.list_collections():
            collections.append({"name": i.name, "count": i.count()})
        
        return collections
    
    ## get documents in specified collection
    def get_collection_data(self, collection_name, dataframe=False, limit=10):
        collection = self.client.get_collection(name=collection_name)
        count = collection.count()
        data = collection.get(limit=limit, offset=0)
        if dataframe:
            return pd.DataFrame(data)
        return data
    
    def get_file_names(self, collection_name):
        collection = self.client.get_collection(name=collection_name)
        all_metadatas  = collection.get(include=["metadatas"]).get('metadatas')
        distinct_keys = set([x.get('file_name') for x  in all_metadatas])
        return {os.path.basename(x) for x in distinct_keys}
    
    ## query specified collection
    def query(self, query_str, collection_name, model_name, k=3, dataframe=False):
        collection = self.client.get_collection(collection_name)
        
        embed_model = HuggingFaceEmbedding(model_name=model_name)
        embedding = embed_model.get_text_embedding(query_str)
        res = collection.query(
            query_embeddings=[embedding], n_results=k
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
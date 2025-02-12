import os
import chromadb 
import pandas as pd

class ChromaDb:
    def __init__(self, path):
        if path is None:
            self.client = chromadb.EphemeralClient()
        else:
            self.client = chromadb.PersistentClient(path)

    ## get list of collections
    def get_collections(self) -> list[dict]:
        collections = []
        collection_names = self.client.list_collections()
        for name in collection_names:
            collection = self.client.get_collection(name)
            collections.append({"name": name, "count": collection.count()})
        
        return collections
    
    ## get documents in specified collection
    def get_collection_data(self, collection_name, dataframe=False, limit=10):
        collection = self.client.get_collection(name=collection_name)
        data = collection.get(limit=limit, offset=0)

        # following lines are to handle None values in the data, chromadb started retruning non-aligned arrays - this is a workaround
        cleaned_data = {key: (value if value is not None else []) for key, value in data.items()}
        if dataframe:
            df = pd.DataFrame.from_dict(cleaned_data, orient='index')
            return df.transpose()
        return cleaned_data
    
    def get_file_names(self, collection_name):
        collection = self.client.get_collection(name=collection_name)
        all_metadatas  = collection.get(include=["metadatas"]).get('metadatas')
        distinct_keys = set([x.get('file_name') for x  in all_metadatas])
        return {os.path.basename(x) for x in distinct_keys}
    
    ## query specified collection
    def query(self, query_str, collection_name, embedding_model, n_result_count=3, dataframe=False):
        collection = self.client.get_collection(collection_name)       

        embedding = embedding_model.get_text_embedding(query_str)
        res = collection.query(
            query_embeddings=[embedding], n_results=n_result_count
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

    def create_collection(self, collectionName):  
        return self.client.get_or_create_collection(collectionName)  
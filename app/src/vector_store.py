import os
import time
import chromadb 
import pandas as pd
from src.embedding_model_factory import EmbeddingModelFactory
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core import StorageContext, VectorStoreIndex, Settings
from llama_index.core.retrievers import AutoMergingRetriever
from llama_index.core.schema import Node, NodeRelationship

class ChromaDb:
    def __init__(self, path=None):
        if path is None:
            self.dbPath = None
            self.client = chromadb.EphemeralClient()
        else:
            self.dbPath = path
            self.client = chromadb.PersistentClient(path)

    ## get list of collections
    def get_collections(self) -> list[dict]:
        collections = []
        collection_names = self.client.list_collections()
        for name in collection_names:
            collection = self.client.get_collection(name)
            collection_data = {"name": name, "count": collection.count()}
            
            for field in ['created_at', 'modified_at']:
                collection_data[field] = collection.metadata.get(field, 0) if collection.metadata else 0
            
            collections.append(collection_data)
        
        return collections
    
    ## get documents in specified collection
    def get_collection_data(self, collection_name, dataframe=False, limit=10):
        collection = self.client.get_collection(name=collection_name)
        data = collection.get(limit=limit, offset=0)

        # following lines are to handle None values in the data, chromadb started returning non-aligned arrays - this is a workaround
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
    
    def query_vector_store_index(self, query_str, collection_name, embedding_model_requested, n_result_count=3, dataframe=False):
        Settings.embed_model = embedding_model_requested
        docstore_path = os.path.join(self.dbPath, "../docstore")
        storage_context = StorageContext.from_defaults(persist_dir=docstore_path)        
        index = self.get_vector_store_index(collection_name)
        
        print(f"Total docs in docstore: {len(storage_context.docstore.docs)}")
        
        base_retriever = index.as_retriever(similarity_top_k=n_result_count)
        auto_merging_retriever = AutoMergingRetriever(
            base_retriever, 
            storage_context,
            verbose=True
        )
        
        ## todo - an exception here crashes the app
        retrieved_nodes = auto_merging_retriever.retrieve(query_str)

        out = {}
        out['documents'] = [node.text for node in retrieved_nodes]
        out['distances'] = [node.score for node in retrieved_nodes]
        out['metadatas'] = [node.metadata for node in retrieved_nodes]
        if dataframe:
            return pd.DataFrame(out)
        return out

        
    ## query specified collection
    def query_collection(self, query_str, collection_name, embedding_model_requested, n_result_count=3, dataframe=False):
        collection = self.client.get_collection(collection_name)
        
        # Get all unique embedding models used in this collection
        embedding_model = EmbeddingModelFactory.get_embedding_model(collection, embedding_model_requested)

        # Generate embeddings for the query
        query_embedding = embedding_model.get_text_embedding(query_str)
        
        # Query the collection
        res = collection.query(
            query_embeddings=[query_embedding], n_results=n_result_count
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
        timestamp = time.time()
        return self.client.get_or_create_collection(collectionName, metadata={"created_at": timestamp, "modified_at": timestamp})  
    
    def update_collection_modified_timestamp(self, collection_name):
        """Update the modified_at timestamp for a collection to the current time"""
        collection = self.client.get_collection(name=collection_name)
        current_metadata = collection.metadata or {}
        # Preserve created_at and other metadata while updating modified_at
        updated_metadata = {**current_metadata, "modified_at": time.time()}
        collection.modify(metadata=updated_metadata)
        return collection
    
    def get_vector_store_index(self, collection_name):
        collection = self.client.get_collection(name=collection_name)
        vector_store = ChromaVectorStore(chroma_collection=collection)

        ## self._write_collection_metadata(collection.get(), collection.name)
        
        # Create index with all nodes
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store
        )
        
        return index        

    def _write_collection_metadata(self, collection_data, collection_name):
        """Write document IDs and metadata from a collection to a text file.
        
        Args:
            collection_data: Dictionary containing collection documents and metadata
            collection_name: Name of the collection to use in output filename
        """
        for i, doc_id in enumerate(collection_data.get('ids', [])):
            metadata = collection_data.get('metadatas', [])[i] or {}
            # Write doc ID and metadata to text file  
            with open(f"{collection_name}_nodes.txt", "a") as f:
                f.write(f"Document ID: {doc_id}\n")
                f.write("Metadata:\n")
                for key, value in metadata.items():
                    f.write(f"  {key}: {value}\n")
                f.write("\n")  # Add blank line between entries
    
    def get_collection_metadata(self, collection_name):
        try:
            collection = self.client.get_collection(name=collection_name)
            metadata = collection.metadata
            if metadata is None:
                metadata = {}
            return metadata
        except Exception as e:
            print(f"Error getting metadata for collection {collection_name}: {e}")
            return {}
    
    
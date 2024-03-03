import logging
import sys
import os

import openai

import torch
import torch.cuda

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader,ServiceContext        
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.ingestion import IngestionPipeline

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

openai.api_key=os.getenv("OPENAI_API_KEY")

class Loader:
    def __init__(self, dbPath):
        self.dbPath = dbPath

        # Check if CUDA is available
        if torch.cuda.is_available():

            # Set the CUDA device
            torch.cuda.set_device(0)
            
            # Get the current device
            device = torch.cuda.device(device=0)

            # Create a tensor on the GPU
            tensor = torch.cuda.FloatTensor([1, 2, 3])

            # Print the tensor
            print(tensor)

        print(torch.cuda.is_available())        

    def load(self, filePath, collectionName):   

        book = SimpleDirectoryReader(input_files=[filePath], filename_as_id=True).load_data()
        llm = OpenAI(temperature=0.1, model_name="gpt-3.5-turbo", max_tokens=512)

        text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=20)
        transformations = [text_splitter]
        pipeline = IngestionPipeline(transformations=transformations)
        embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-mpnet-base-v2")
        db = chromadb.PersistentClient(path=self.dbPath)
        chroma_collection = db.get_or_create_collection(collectionName)

        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm, transformations=transformations)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(book, storage_context=storage_context, service_context=service_context)
        print(index)        
import logging
import sys
import os

import torch
import torch.cuda
from openai import OpenAI
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader,ServiceContext        
from llama_index.core import StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

class Loader:
    def __init__(self, dbPath):
        self.dbPath = dbPath

        # Check if CUDA is available
        print(f'CUDA is available: {torch.cuda.is_available()}')  
        if torch.cuda.is_available():

            # Set the CUDA device
            torch.cuda.set_device(0)
            
            # Get the current device
            device = torch.cuda.device(device=0)

            # Create a tensor on the GPU
            tensor = torch.tensor([1, 2, 3])

            # Print the tensor
            print(tensor)      

    def load(self, db, filePath, collectionName, embeddingModelName, inferenceModelName, apiKey, apiBaseUrl, useExtractors=False, temperature=0.1, timeout=30):   

        book = SimpleDirectoryReader(input_files=[filePath], filename_as_id=True).load_data()
        
        ## todo: extract openai initialization to a separate class
        llm = OpenAI(model=inferenceModelName, 
                     api_base=apiBaseUrl, 
                     api_key=apiKey, 
                     max_tokens=1024, 
                     temperature=temperature, 
                     timeout=timeout)

        text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=20)
        extractors=[
            TitleExtractor(nodes=1, llm=llm, num_workers=1) #title is located on the first page, so pass 1 to nodes param
            ,QuestionsAnsweredExtractor(questions=3, llm=llm, num_workers=1) #let's extract 3 questions for each node, you can customize this.
            ,SummaryExtractor(summaries=["self"], llm=llm, num_workers=1) #let's extract the summary for both previous node and current node.
            ,KeywordExtractor(keywords=10, llm=llm, num_workers=1) #let's extract 10 keywords for each node.
        ]                
        transformations = [text_splitter] + extractors if useExtractors else [text_splitter]

        ## todo: extract embedding initialization to a separate class
        if (embeddingModelName == "OpenAIEmbedding"):
            embed_model = OpenAIEmbedding(api_base=apiBaseUrl, 
                                          api_key=apiKey, 
                                          timeout=timeout, 
                                          embed_batch_size=100)
        else:
            embed_model = HuggingFaceEmbedding(
                model_name=embeddingModelName,
                embed_batch_size=100,
                device='cuda' if torch.cuda.is_available() else 'cpu'
                )        

        chroma_collection = db.create_collection(collectionName)

        service_context = ServiceContext.from_defaults(embed_model=embed_model, llm=llm, transformations=transformations)
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(book, storage_context=storage_context, service_context=service_context)
        print(index)        
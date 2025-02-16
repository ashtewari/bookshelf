import logging
import sys
import os

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import SimpleDirectoryReader, StorageContext
from llama_index.core.text_splitter import TokenTextSplitter
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.readers.docling import DoclingReader
from llama_index.core import Settings

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

class Loader:
    def __init__(self, dbPath):
        self.dbPath = dbPath

    def load(self, db, filePath, collectionName, embedding_model, llm, useExtractors=False):   

        reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
        node_parser = DoclingNodeParser()
        docs = SimpleDirectoryReader(input_files=[filePath], filename_as_id=True, file_extractor={"*.*": reader},).load_data()

        text_splitter = TokenTextSplitter(separator=" ", chunk_size=512, chunk_overlap=20)
        extractors=[
            TitleExtractor(nodes=1, llm=llm, num_workers=1) #title is located on the first page, so pass 1 to nodes param
            ,QuestionsAnsweredExtractor(questions=3, llm=llm, num_workers=1) #let's extract 3 questions for each node, you can customize this.
            ,SummaryExtractor(summaries=["self"], llm=llm, num_workers=1) #let's extract the summary for both previous node and current node.
            ,KeywordExtractor(keywords=10, llm=llm, num_workers=1) #let's extract 10 keywords for each node.
        ]                
        transformations = [node_parser] + extractors if useExtractors else [text_splitter]

        chroma_collection = db.create_collection(collectionName)

        # Update Settings instead of using ServiceContext
        Settings.embed_model = embedding_model
        Settings.llm = llm
        Settings.transformations = transformations

        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        print(index)        
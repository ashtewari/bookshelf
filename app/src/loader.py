import logging
import sys
import os

from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.storage.docstore import SimpleDocumentStore
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
from src.embedding_model_factory import EmbeddingModelFactory
from src.chunking_strategy import TokenTextSplitterStrategy, DoclingNodeParserStrategy, HierarchicalNodeParserStrategy, SemanticSplitterNodeParserStrategy
from llama_index.core.node_parser import HierarchicalNodeParser, SemanticSplitterNodeParser
from llama_index.core.node_parser import get_leaf_nodes, get_root_nodes

logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

os.environ['NUMEXPR_MAX_THREADS'] = '4'
os.environ['NUMEXPR_NUM_THREADS'] = '2'

class Loader:
    def __init__(self, dbPath):
        self.dbPath = dbPath
        self.chunking_strategy = TokenTextSplitterStrategy()  # default strategy

    def set_transformer_strategy(self, strategy):
        self.chunking_strategy = strategy

    def load_documents(self, filePath):
        if isinstance(self.chunking_strategy, DoclingNodeParserStrategy):
            reader = DoclingReader(export_type=DoclingReader.ExportType.JSON)
            return SimpleDirectoryReader(
                input_files=[filePath], 
                filename_as_id=True, 
                file_extractor={".pdf": reader}
            ).load_data()
        else:
            return SimpleDirectoryReader(
                input_files=[filePath], 
                filename_as_id=True
            ).load_data()

    def load(self, db, filePath, collectionName, embedding_model_requested, llm, useExtractors=False):   
        docs = self.load_documents(filePath)

        # Get all unique embedding models used in this collection
        collection_data = db.get_collection_data(collectionName)
        embedding_model = EmbeddingModelFactory.get_embedding_model(collection_data, embedding_model_requested)

        # Add embedding model info to document metadata
        for doc in docs:
            if not doc.metadata:
                doc.metadata = {}
            doc.metadata["embedding_model"] = embedding_model.model_name

        # Get base transformations (chunking) from strategy
        node_parser = None
        if isinstance(self.chunking_strategy, SemanticSplitterNodeParserStrategy):
            node_parser = self.chunking_strategy.get_transformation(llm, embedding_model)
        else:
            node_parser = self.chunking_strategy.get_transformation(llm)

        # Optionally add extractors for any strategy
        extractors = []
        if useExtractors:
            extractors = [
                TitleExtractor(nodes=1, llm=llm, num_workers=1),
                QuestionsAnsweredExtractor(questions=3, llm=llm, num_workers=1),
                SummaryExtractor(summaries=["self"], llm=llm, num_workers=1),
                KeywordExtractor(keywords=10, llm=llm, num_workers=1),
            ]

        chroma_collection = db.create_collection(collectionName)

        # Update Settings instead of using ServiceContext
        Settings.embed_model = embedding_model
        Settings.llm = llm

        if isinstance(self.chunking_strategy, HierarchicalNodeParserStrategy):
            transformations = extractors
            Settings.transformations = transformations

            # Get all nodes including the hierarchy
            all_nodes = node_parser[0].get_nodes_from_documents(docs)
            leaf_nodes = get_leaf_nodes(all_nodes)
            root_nodes = get_root_nodes(all_nodes)
            
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            docstore_path = os.path.join(self.dbPath, "../docstore")
            os.makedirs(docstore_path, exist_ok=True)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)

            # First add ALL nodes to docstore using add_documents
            storage_context.docstore.add_documents(all_nodes, allow_update=True)
            storage_context.persist(persist_dir=docstore_path)
            
            # Then create index with leaf nodes
            index = VectorStoreIndex(
                nodes=leaf_nodes,
                storage_context=storage_context,
                show_progress=True
            )
            
            # Verify nodes are stored
            print(f"Total nodes in docstore: {len(storage_context.docstore.docs)}")
            print(f"Leaf nodes: {len(leaf_nodes)}")
            print(f"Root nodes: {len(root_nodes)}")
        else:
            transformations = node_parser + extractors
            Settings.transformations = transformations
            vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex.from_documents(docs, storage_context=storage_context)
        
        print(index)        
from abc import ABC, abstractmethod
from llama_index.core.text_splitter import TokenTextSplitter, SentenceSplitter
from llama_index.node_parser.docling import DoclingNodeParser
from llama_index.core.extractors import (
    SummaryExtractor,
    QuestionsAnsweredExtractor,
    TitleExtractor,
    KeywordExtractor,
)

class ChunkingStrategy(ABC):
    @abstractmethod
    def get_transformation(self, llm=None):
        pass
    
    @abstractmethod
    def get_config_options(self):
        """Return a dictionary of configuration options and their default values"""
        pass
    
    @abstractmethod
    def update_config(self, config):
        """Update the strategy configuration"""
        pass

class TokenTextSplitterStrategy(ChunkingStrategy):
    def __init__(self, chunk_size=512, chunk_overlap=20, separator=" "):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def get_transformation(self, llm=None):
        text_splitter = TokenTextSplitter(
            separator=self.separator,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return [text_splitter]
    
    def get_config_options(self):
        return {
            "chunk_size": {"value": self.chunk_size, "min": 100, "max": 2048, "step": 32},
            "chunk_overlap": {"value": self.chunk_overlap, "min": 0, "max": 200, "step": 10},
            "separator": {"value": self.separator, "options": [" ", "\n", ".", ","]}
        }
    
    def update_config(self, config):
        self.chunk_size = config.get("chunk_size", self.chunk_size)
        self.chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)
        self.separator = config.get("separator", self.separator)

class SentenceSplitterStrategy(ChunkingStrategy):
    def __init__(self, chunk_size=1024, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def get_transformation(self, llm=None):
        sentence_splitter = SentenceSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        return [sentence_splitter]
    
    def get_config_options(self):
        return {
            "chunk_size": {"value": self.chunk_size, "min": 100, "max": 2048, "step": 32},
            "chunk_overlap": {"value": self.chunk_overlap, "min": 0, "max": 200, "step": 10}
        }
    
    def update_config(self, config):
        self.chunk_size = config.get("chunk_size", self.chunk_size)
        self.chunk_overlap = config.get("chunk_overlap", self.chunk_overlap)

class DoclingNodeParserStrategy(ChunkingStrategy):
    def __init__(self, num_workers=1):
        self.num_workers = num_workers

    def get_transformation(self, llm=None):
        node_parser = DoclingNodeParser()
        return [node_parser]
    
    def get_config_options(self):
        return {
            "num_workers": {"value": self.num_workers, "min": 1, "max": 8, "step": 1}
        }
    
    def update_config(self, config):
        self.num_workers = config.get("num_workers", self.num_workers) 
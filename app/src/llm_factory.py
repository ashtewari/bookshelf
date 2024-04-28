# llm_factory.py
from openai import OpenAI as oai
from llama_index.llms.openai import OpenAI as llama_oai

class llm:
    @staticmethod
    def create_instance_for_extraction(model, api_base, api_key, max_tokens, temperature, timeout):
        return llama_oai(model=model, api_base=api_base, api_key=api_key, max_tokens=max_tokens, temperature=temperature, timeout=timeout)
    
    @staticmethod
    def create_instance_for_inference(api_base, api_key, timeout):
        return oai(base_url=api_base, api_key=api_key, timeout=timeout)    
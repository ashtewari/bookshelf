# llm_factory.py

from llama_index.llms.openai import OpenAI as llama_oai

class llm:
    def create_instance_for_extraction(self, model, api_base, api_key, max_tokens, temperature, timeout):
        return llama_oai(model=model, api_base=api_base, api_key=api_key, max_tokens=max_tokens, temperature=temperature, timeout=timeout)   

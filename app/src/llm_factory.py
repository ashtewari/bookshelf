# llm_factory.py

from langchain_openai import ChatOpenAI as langchain_llm
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage
from llama_index.llms.openai import OpenAI as llama_oai

class llm:
    def create_instance_for_extraction(self, model, api_base, api_key, max_tokens, temperature, timeout):
        return llama_oai(model=model, api_base=api_base, api_key=api_key, max_tokens=max_tokens, temperature=temperature, timeout=timeout)   
    
    def execute_prompt(self, api_base, api_key, model_name, timeout, prompt, temperature):
        llm = langchain_llm(openai_api_base=f"{api_base}", openai_api_key=api_key, model=model_name, request_timeout=timeout) 
        messages = [HumanMessage(content=[{"type": "text", "text": str(prompt)},])]      
        response = llm.invoke(input=messages)
        return response.content
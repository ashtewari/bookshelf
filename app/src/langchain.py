# langchain.py

from langchain.llms import OpenAI as OpenAITextGen
from langchain_openai import ChatOpenAI 
from langchain_core.messages import HumanMessage

class llm_openai:

    def execute_prompt(self, api_base, api_key, model_name, timeout, prompt, temperature):
        llm = ChatOpenAI(openai_api_base=f"{api_base}", openai_api_key=api_key, model=model_name, request_timeout=timeout) 
        kwargs={
                "temperature": temperature, 
                "max_tokens": 1000,
                "presence_penalty":0.0,
                "frequency_penalty":0.0,
                "n":1,
                "top_p":1.0                                    
                }        
        
        # using invoke method
        response = llm.invoke(input=[prompt,], **kwargs)
        return response.content
        
        """
        # using generate method
        messages = [HumanMessage(content=[{"type": "text", "text": str(prompt)},])] 
        response = llm.generate(messages=[messages], **kwargs)
        return response.generations[0][0].text 
        """  
        
    def execute_instruct_prompt(self, api_key, prompt):
        ## using default instruct model
        llm = OpenAITextGen(openai_api_key=api_key)
        response = llm.generate([prompt])
        return response.generations[0][0].text
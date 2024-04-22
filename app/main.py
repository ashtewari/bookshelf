import os
import sys
import streamlit as st
import uuid
import re
from dotenv import load_dotenv, find_dotenv
from src.viewer import ChromaDb
import tempfile
from src.loader import Loader
from openai import OpenAI
import pandas as pd
import json

load_dotenv(find_dotenv(), override=True) 

def main():
    st.set_page_config(page_title="Bookshelf", page_icon="📚", layout="wide")

    st.title("Bookshelf", "📚")
    configure_settings()

    preferred_data_path = None
    timeout = 30
    if 'Bookshelf:PreferredDataPath' in os.environ:
        preferred_data_path = os.environ['Bookshelf:PreferredDataPath']
    if 'OPENAI_API_TIMEOUT' in os.environ:
        timeout = os.getenv('OPENAI_API_TIMEOUT')        

    temp_dir = os.path.join(tempfile.gettempdir(), "bookshelf") 
    app_user_data_path = os.path.join(temp_dir, os.path.join("data", "db"))        
    data_path = app_user_data_path  
 
    if(preferred_data_path is not None):
        data_path = st.text_input("Data Path", placeholder="Full path to database directory", value=preferred_data_path)
    
    if (data_path==""):
        st.error("Please provide a valid data path")
    
    collection_selected = st.session_state.collections if 'collections' in st.session_state else None
    collection_name = st.text_input("Collection Name", key="specified_collection_name", value=collection_selected["name"] if collection_selected else "default")
    
    db = ChromaDb(data_path)    
    db.client.create_collection(collection_name, get_or_create=True)

    uploaded_file = None
    submitted = False
    use_extractors = False
    with st.form("frmFileUploader", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose File")
        use_extractors = st.radio("Extract additional metadata", options=[False, True], index=0, format_func=lambda x: "Yes" if x else "No")
        submitted = st.form_submit_button("Upload File")
    if submitted and uploaded_file is not None:
           
        tempFilePath = os.path.join(temp_dir, uploaded_file.name)
        with open(tempFilePath, "wb") as f:
                f.write(uploaded_file.getvalue())       
        print(f"Selected file: {tempFilePath}")

        loader = Loader(data_path)
        collection_name = collection_name or os.path.basename(st.session_state.embedding_model_name)
        with st.spinner(f"Uploading {uploaded_file.name} [{use_extractors}] to {collection_name}"):
            loader.load(filePath=tempFilePath
                        , collectionName=generate_valid_collection_name(collection_name)
                        , embeddingModelName=st.session_state.embedding_model_name
                        , inferenceModelName=st.session_state.inference_model_name
                        , apiKey=st.session_state.api_key
                        , apiBaseUrl=st.session_state.api_url
                        , useExtractors=use_extractors
                        , temperature=0.1 
                        , timeout=timeout)
            os.remove(tempFilePath)
      
    with st.spinner("Loading collections..."):
        collections = db.get_collections()
    names = [d['name'] for d in collections]
    collection_index = names.index(collection_name) if collection_name in names else 0
    col1, col2 = st.columns([1,3])
    with col1:
        collection_selected=st.radio("Collections", key="collections",
                options=collections, format_func=lambda x: x['name'],
                index=collection_index,
                )
        st.button(f"Delete selected collection: {collection_selected["name"]}", on_click=lambda: db.delete_collection(collection_selected["name"]))
    with col2:         
        if collection_selected: 
            limit = st.slider('Chunks', 1, collection_selected["count"], 10, )
            with st.spinner(f"Loading {collection_selected}..."):
                df = db.get_collection_data(collection_selected["name"], dataframe=True, limit=limit)
                st.dataframe(df, use_container_width=True, height=300) 
            with st.spinner(f"Loading {collection_selected}..."):
                file_names = db.get_file_names(collection_selected["name"])          
                st.dataframe(file_names, use_container_width=True, column_config={'value': st.column_config.TextColumn(label='Files')}) 
    st.divider()

    query = st.text_input("Find similar text"
                          , key="txtFindText"
                          , placeholder="Enter text to search")
    result_count = st.number_input("Number of chunks to find", value=5, format='%d', step=1)
    with st.form(key="frmQuery", clear_on_submit=True, border=False):
        submittedSearch = st.form_submit_button("Search")

    if submittedSearch and query is not None and query != "":
        if result_count == '':
            result_count = 5  # Set a default value if result_count is empty

        with st.spinner(f"Searching for simmilar documents ..."):
            result_df = db.query(query, collection_selected["name"], st.session_state.embedding_model_name, int(result_count), dataframe=True)
    
        st.dataframe(result_df, use_container_width=True)
        result_df['metadatas'] = result_df['metadatas'].apply(lambda x: json.dumps(x) if not isinstance(x, dict) else x)
        result_df['file_name'] = pd.json_normalize(result_df['metadatas'])['file_name'].apply(lambda x: os.path.basename(x))
        result_df['page_label'] = pd.json_normalize(result_df['metadatas'])['page_label']
        st.table(result_df.groupby('file_name')['page_label'].apply(list).apply(set).apply(sorted))
    
        context = result_df['documents'].to_list() 
        prompt = f"CONTEXT = {context} *** \n Based on the CONTEXT provided above, {query}"
        
        with st.spinner("Thinking ..."):
            llm_response = get_completion(prompt, model=st.session_state.inference_model_name, temperature=0.2, timeout=timeout)
        
        st.text_area(key="txtLlmResponse", label=query, value=llm_response)   

def configure_settings():
    key_choice = st.sidebar.radio(label="LLM Settings", options=("OpenAI", "Local"), horizontal=True)

    api_key = "not-needed"
    api_url = "http://localhost:1234/v1"
    embedding_model_name = "sentence-transformers/all-mpnet-base-v2"
    inference_model_name = "gpt-3.5-turbo"
    
    if key_choice == "OpenAI":
        embedding_model_name = "OpenAIEmbedding"
        api_url_value = "https://api.openai.com/v1"
        inference_model_name = st.sidebar.text_input(key="txtInferenceModelName", label="Model Name", value=inference_model_name)               
    elif key_choice == "Local":
        api_url_value = "http://localhost:1234/v1"
        embedding_model_name = st.sidebar.text_input(key="txtEmbeddingModelName", label="Embedding Model Name", placeholder="sentence-transformers/all-MiniLM-L6-v2", value=embedding_model_name)
    
    api_url = st.sidebar.text_input(key="txtApiUrl", label="LLM API Url", placeholder="https://api.openai.com/v1", value=api_url_value)
    api_key = st.sidebar.text_input(key="txtApiKey", label="API Key", type="password", value=os.getenv('OPENAI_API_KEY') or "not-needed")
    
    st.session_state.api_key = api_key
    st.session_state.api_url = api_url
    st.session_state.embedding_model_name = embedding_model_name
    st.session_state.inference_model_name = inference_model_name        

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0, timeout=30): 
    messages = [{"role": "user", "content": prompt}]
    client = OpenAI()
    client.base_url = st.session_state.api_url 
    client.api_key = st.session_state.api_key
    client.timeout = timeout
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content

def generate_valid_collection_name(input):
    #Expected collection name that 
    #(1) contains 3-63 characters, 
    #(2) starts and ends with an alphanumeric character, 
    #(3) otherwise contains only alphanumeric characters, underscores or hyphens (-), 
    #(4) contains no two consecutive periods
    
    # minimum 3 characters
    result = input
    if len(input) < 3:
        result = input + "-" + str(uuid.uuid4())
    
    # remove spaces
    result = result.replace(" ", "_")
    # remove all non-alphanumeric characters
    result = ''.join(e for e in result if e.isalnum() or e == "_" or e == "-")
    # remove consecutive periods
    result = re.sub(r'\.{2,}', '-', result)
    # max 63 characters
    result =  result[0:63].lower()
    
    return result 
         
if __name__ == "__main__":

    working_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(working_dir)

    main()

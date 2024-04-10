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

    preferred_data_path = None
    timeout = 30
    if 'Bookshelf:PreferredDataPath' in os.environ:
        preferred_data_path = os.environ['Bookshelf:PreferredDataPath']
    if 'OPENAI_API_TIMEOUT' in os.environ:
        timeout = os.getenv('OPENAI_API_TIMEOUT')        

    collection_selected = st.session_state.collections if 'collections' in st.session_state else None
    data_path = st.text_input("Data Path", placeholder="Full path to database directory", value=preferred_data_path or "data")
    if (data_path==""):
        st.error("Please provide a valid data path")
    
    model_name = st.text_input("Embedding Model Name", placeholder="sentence-transformers/all-MiniLM-L6-v2", value="sentence-transformers/all-mpnet-base-v2")
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
           
        temp_dir = tempfile.mkdtemp()
        tempFilePath = os.path.join(temp_dir, uploaded_file.name)
        with open(tempFilePath, "wb") as f:
                f.write(uploaded_file.getvalue())       
        print(f"Selected file: {tempFilePath}")

        loader = Loader(data_path)
        collection_name = collection_name or os.path.basename(model_name)
        with st.spinner(f"Uploading {uploaded_file.name} [{use_extractors}] to {collection_name}"):
            loader.load(tempFilePath, generate_valid_collection_name(collection_name), model_name, useExtractors=use_extractors, timeout=timeout)
            os.remove(tempFilePath)
      
    with st.spinner("Loading collections..."):
        collections = db.get_collections()
    names = {k for d in collections for k in d.keys()}
    collection_index = collections.index(collection_name) if collection_name in names else 0
    col1, col2 = st.columns([1,3])
    with col1:
        collection_selected=st.radio("Collections", key="collections",
                options=collections, format_func=lambda x: x['name'],
                index=collection_index,
                on_change=lambda : st.session_state.update(txtFindText=None)
                )
        st.button(f"Delete selected collection: {collection_selected["name"]}", on_click=lambda: db.delete_collection(collection_selected["name"]))
    with col2:         
        if collection_selected: 
            limit = st.slider('Select nodes', 1, collection_selected["count"], 10, )
            with st.spinner(f"Loading {collection_selected}..."):
                result_df = db.get_collection_data(collection_selected["name"], dataframe=True, limit=limit)                  
                result_df['metadatas'] = result_df['metadatas'].apply(lambda x: json.dumps(x) if not isinstance(x, dict) else x)
                result_df['file_name'] = pd.json_normalize(result_df['metadatas'])['file_name'].apply(lambda x: os.path.basename(x))           
                st.dataframe(result_df, use_container_width=True, height=300) 
                st.dataframe(result_df['file_name'].drop_duplicates(), hide_index=True, use_container_width=True) 
    
    st.divider()

    query = st.text_input("Find similar text", key="txtFindText", placeholder="Enter text to search")
    result_count = st.number_input("Number of documents to find", value=5, format='%d', step=1)
    if query:
        if result_count == '':
            result_count = 5  # Set a default value if result_count is empty

        with st.spinner(f"Searching for simmilar documents ..."):
            result_df = db.query(query, collection_selected["name"], model_name, int(result_count), dataframe=True)
    
        st.dataframe(result_df, use_container_width=True)
        result_df['metadatas'] = result_df['metadatas'].apply(lambda x: json.dumps(x) if not isinstance(x, dict) else x)
        result_df['file_name'] = pd.json_normalize(result_df['metadatas'])['file_name'].apply(lambda x: os.path.basename(x))
        result_df['page_label'] = pd.json_normalize(result_df['metadatas'])['page_label']
        st.table(result_df.groupby('file_name')['page_label'].apply(list).apply(set).apply(sorted))
    
        context = result_df['documents'].to_list() 
        prompt = f"CONTEXT = {context} *** \n Based on the CONTEXT provided above, {query}"
        
        with st.spinner("Thinking ..."):
            llm_response = get_completion(prompt, model="gpt-3.5-turbo", temperature=0.2, timeout=timeout)
        
        st.text_area(key="txtLlmResponse", label=query, value=llm_response)   

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0, timeout=30): 
    messages = [{"role": "user", "content": prompt}]
    client = OpenAI()
    client.base_url = os.getenv('OPENAI_API_URL') 
    client.api_key = os.getenv('OPENAI_API_KEY')
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

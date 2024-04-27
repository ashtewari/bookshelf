import os
import sys
import streamlit as st
import uuid
import re
from dotenv import load_dotenv, find_dotenv
from openai import OpenAI
import pandas as pd
import json
import platform
if platform.system() == 'Linux':
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
from src.viewer import ChromaDb
from src.loader import Loader

load_dotenv(find_dotenv(), override=True) 
# os.environ['OPENAI_API_KEY'] = ""

def main():
    st.set_page_config(page_title="Bookshelf", page_icon="📚", layout="wide")

    st.title("Bookshelf", "📚")

    configure_demo_mode()
    print(f"Demo mode: {st.session_state.demo_mode}") 

    configure_settings(st.session_state.demo_mode)

    preferred_data_path = None
    timeout = 30
    if 'Bookshelf:PreferredDataPath' in os.environ:
        preferred_data_path = os.environ['Bookshelf:PreferredDataPath']
    if 'OPENAI_API_TIMEOUT' in os.environ:
        timeout = os.getenv('OPENAI_API_TIMEOUT')        

    temp_dir = os.path.join("tmp", "bookshelf") 
    app_user_data_path = os.path.join(temp_dir, os.path.join("data", "db"))        
    data_path = app_user_data_path  
 
    if(preferred_data_path is not None):
        data_path = st.text_input("Data Path", placeholder="Full path to database directory", value=preferred_data_path)
    
    if (data_path==""):
        st.error("Please provide a valid data path")
    
    os.makedirs(data_path, exist_ok=True)
    collection_selected = st.session_state.collections if 'collections' in st.session_state else None
    collection_name = st.text_input("Collection Name", key="specified_collection_name", value=collection_selected["name"] if collection_selected else "default")
    
    db = OpenDbConnection(data_path if st.session_state.demo_mode != "1" else None)    
    db.create_collection(collection_name)

    uploaded_file = None
    submitted = False
    use_extractors = False
    use_extractors = st.radio(key="rdUseExtractors", label="Extract additional metadata when loading files?", 
                                options=[False, True], index=0, horizontal=True, 
                                format_func=lambda x: "Yes" if x else "No")    
    if "rdUseExtractors" in st.session_state and st.session_state.rdUseExtractors == True:
        st.warning("WARNING: Metadata extraction uses LLM and may incur additional costs.") 
    
    with st.form("frmFileUploader", clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose File", accept_multiple_files=True, type=["pdf", "docx", "txt", "doc", "log"])
        submitted = st.form_submit_button("Upload File", disabled=st.session_state.api_key_is_valid is False)
    if submitted and uploaded_file is not None:
        for i in range(len(uploaded_file)):
                    
            tempFilePath = os.path.join(temp_dir, uploaded_file[i].name)
            with open(tempFilePath, "wb") as f:
                    f.write(uploaded_file[i].getvalue())       
            print(f"Selected file: {tempFilePath}")

            loader = Loader(data_path)
            collection_name = collection_name or os.path.basename(st.session_state.embedding_model_name)
            with st.spinner(f"Uploading {uploaded_file[i].name} [{use_extractors}] to {collection_name}"):
                loader.load(db=db, filePath=tempFilePath
                        , collectionName=generate_valid_collection_name(collection_name)
                        , embeddingModelName=st.session_state.embedding_model_name
                        , inferenceModelName=st.session_state.inference_model_name
                        , apiKey=st.session_state.api_key
                        , apiBaseUrl=st.session_state.api_url
                        , useExtractors=use_extractors
                        , temperature=0.1 
                        , timeout=int(timeout))
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
        st.button(f"Delete selected collection: {collection_selected['name']}", on_click=lambda: db.delete_collection(collection_selected["name"]))
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

    query = st.text_input("Find similar text chunks"
                          , key="txtFindText"
                          , placeholder="Enter text to search")
    result_count = st.number_input("Number of chunks to find", value=5, format='%d', step=1)
    with st.form(key="frmQuery", clear_on_submit=True, border=False):
        submittedSearch = st.form_submit_button("Search", disabled=st.session_state.api_key_is_valid is False)

    if submittedSearch and query is not None and query != "":
        if result_count == '':
            result_count = 5  # Set a default value if result_count is empty

        with st.spinner(f"Searching for similar documents ..."):
            result_df = db.query(query, collection_selected["name"]
                                , apiKey=st.session_state.api_key
                                , apiBaseUrl=st.session_state.api_url
                                , timeout=int(timeout)                                  
                                , embedding_model_name=st.session_state.embedding_model_name 
                                , k = int(result_count)
                                , dataframe=True)
    
        st.session_state.result_df = result_df
    
    result_df = st.session_state.result_df if 'result_df' in st.session_state else None
    if result_df is not None:
        st.dataframe(result_df, use_container_width=True)
        result_df['metadatas'] = result_df['metadatas'].apply(lambda x: json.dumps(x) if not isinstance(x, dict) else x)
        result_df['file_name'] = pd.json_normalize(result_df['metadatas'])['file_name'].apply(lambda x: os.path.basename(x))
        result_df['page_label'] = pd.json_normalize(result_df['metadatas'])['page_label']
        st.table(result_df.groupby('file_name')['page_label'].apply(list).apply(set).apply(sorted))
        
    st.divider()
    queryPrompt = st.text_input("Prompt LLM"
                , key="txtPromptQuery"
                , placeholder="Enter prompt for LLM")  
    with st.form(key="frmPromptQuery", clear_on_submit=True, border=False):
        submittedLLMSearch = st.form_submit_button("Search", disabled=st.session_state.api_key_is_valid is False)

        if submittedLLMSearch and queryPrompt is not None and queryPrompt != "":    
            context = st.session_state.result_df['documents'].to_list() 
            prompt = f"CONTEXT = {context} *** \n Based on the CONTEXT provided above, {queryPrompt}"
            
            with st.spinner("Thinking ..."):
                llm_response = get_completion(prompt, model=st.session_state.inference_model_name, temperature=0.2, timeout=timeout)
            
            st.text_area(key="txtLlmResponse", label=queryPrompt, value=llm_response)   

@st.cache_resource
def OpenDbConnection(data_path):
    db = ChromaDb(data_path)
    return db

def configure_settings(demo_mode):
    if 'llm_options' in st.session_state:
        llm_options = st.session_state.llm_options
    else:
        llm_options = ["OpenAI", "Local"]
        st.session_state.llm_options = llm_options

    st.sidebar.header("Settings")
    if demo_mode == "1":
        st.warning("WARNING: Shared database for demo. Do not upload personal documents.") 
    if is_running_in_streamlit_cloud():      
        llm_options = ["OpenAI"]

    key_choice = st.sidebar.radio(key="rdOptions", label="Language Model", options=llm_options, horizontal=True)

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
    api_key = st.sidebar.text_input(key="txtApiKey", label="API Key", type="password")
    
    st.session_state.api_key_is_valid = True
    if key_choice == "OpenAI" and (api_key is None or api_key == ""):
        st.sidebar.error("OpenAI API Key is required.")
        st.session_state.api_key_is_valid = False
    elif key_choice == "Local" and (api_key is None or api_key == ""):
        api_key = "not-set"
    
    st.session_state.api_key = api_key
    st.session_state.api_url = api_url
    st.session_state.embedding_model_name = embedding_model_name
    st.session_state.inference_model_name = inference_model_name        

def configure_demo_mode():
    demo_mode = 0
    if 'Bookshelf_Demo_Mode' in os.environ:
        demo_mode = os.getenv('Bookshelf_Demo_Mode')
    elif is_running_in_streamlit_cloud():
            demo_mode = 1
    st.session_state.demo_mode = demo_mode

def get_completion(prompt, model="gpt-3.5-turbo", temperature=0, timeout=30): 
    messages = [{"role": "user", "content": prompt}]
    client = OpenAI(api_key = st.session_state.api_key)
    client.base_url = st.session_state.api_url 
    client.timeout = int(timeout)
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

def is_running_in_streamlit_cloud():
  """Returns True if the code is running in Streamlit Cloud, False otherwise."""

  if platform.processor():
    return False
  else:
    return True
           
if __name__ == "__main__":

    working_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(working_dir)

    main()

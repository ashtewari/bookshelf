import os
import sys
import streamlit as st
import uuid
import re
from dotenv import load_dotenv, find_dotenv
from src.viewer import ChromaDb
import tempfile
from src.loader import Loader

load_dotenv(find_dotenv()) 

def main():
    st.set_page_config(page_title="bookshelf", page_icon="📚", layout="wide")

    st.title("bookshelf", "📚")

    preferred_data_path = None
    if 'Bookshelf:PreferredDataPath' in os.environ:
        preferred_data_path = os.environ['Bookshelf:PreferredDataPath']

    data_path = st.text_input("Database", placeholder="Full path to database directory", value=preferred_data_path or "data")

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
           
        temp_dir = tempfile.mkdtemp()
        tempFilePath = os.path.join(temp_dir, uploaded_file.name)
        with open(tempFilePath, "wb") as f:
                f.write(uploaded_file.getvalue())       
        st.write(f"Selected file: {tempFilePath}")

        loader = Loader(data_path)
        loader.load(tempFilePath, generate_valid_collection_name(uploaded_file.name))
    
    st.divider()

    if not(data_path==""):
        db = ChromaDb(data_path)

        col1, col2 = st.columns([1,3])
        with col1:
            collection_selected=st.radio("Collections",
                 options=db.get_collections(),
                 index=0,
                 )
        
        with col2:
            if collection_selected:
                df = db.get_collection_data(collection_selected, dataframe=True)

                st.markdown(f"<b>Selected Collection </b>*{collection_selected}*", unsafe_allow_html=True)
                st.dataframe(df, use_container_width=True, height=300)
        
        st.divider()

        query = st.text_input("Find similar text", placeholder="Enter text to search")
        model_name = st.text_input("Enter Embedding Model Name", placeholder="all-MiniLM-L6-v2")
        result_count = st.number_input("Enter number of documents to find", value=5, format='%d', step=1)
        if query:
            if result_count == '':
                result_count = 5  # Set a default value if result_count is empty
            result_df = db.query(query, collection_selected, model_name, int(result_count), dataframe=True)
        
            st.dataframe(result_df, use_container_width=True)   

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
    
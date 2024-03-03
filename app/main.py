import os
import sys
import streamlit as st
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from src.viewer import ChromaDb
import tempfile

load_dotenv(find_dotenv()) 

def main():
    st.set_page_config(page_title="bookshelf", page_icon="📚", layout="wide")

    st.title("bookshelf", "📚")

    last_used_path = os.environ['MBED:SelectedDatabasePath']
    path = st.text_input("Database", placeholder="Full path to database directory", value=last_used_path)

    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
           
        temp_dir = tempfile.mkdtemp()
        tempFilePath = os.path.join(temp_dir, uploaded_file.name)
        with open(tempFilePath, "wb") as f:
                f.write(uploaded_file.getvalue())       
        st.write(f"Selected file: {tempFilePath}")
    
    st.divider()

    if not(path==""):
        db = ChromaDb(path)

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
            
if __name__ == "__main__":

    working_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(working_dir)

    main()
    
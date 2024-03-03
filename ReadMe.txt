## Run Local
# Create virtual environment
conda create --name bookshelf python=3.11
conda activate bookshelf
pip install -r requirements.txt
streamlit run app/main.py

## Run with Docker
# Build the Docker image
docker build -t bookshelf .
# Run the Docker container
docker run -v c:\data\:/app/data -p 8501:8501 bookshelf 
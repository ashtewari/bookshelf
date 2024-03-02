# Start from a base image with Python installed
FROM python:3.11.5-bookworm

# Create a directory for the app and copy the requirements.txt file
RUN mkdir -p /build
COPY requirements.txt /build/

# Install dependencies
RUN pip install -r /build/requirements.txt

# Copy the rest of the app's code
COPY . /build

# Expose port 8501 and run the app
EXPOSE 8501
CMD ["streamlit", "run", "/build/app/main.py"]
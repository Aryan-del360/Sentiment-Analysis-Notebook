# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
# This is crucial for 'vader_lexicon' to be available in the container
RUN python -m nltk.downloader vader_lexicon

# Copy the rest of your application code
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to run your Streamlit app
# --server.port sets the port to 8501 (as exposed above)
# --server.enableCORS false and --server.enableXsrfProtection false are often needed for Cloud Run
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
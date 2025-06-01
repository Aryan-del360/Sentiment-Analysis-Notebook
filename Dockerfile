# Use an official Python runtime as a parent image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (vader_lexicon)
# This is crucial for 'vader_lexicon' to be available in the container
RUN python -m nltk.downloader vader_lexicon

# Copy the rest of your application code
COPY . .

# Expose the port that Streamlit runs on
EXPOSE 8501

# Define the command to run your Streamlit app
# IMPORTANT: Ensure 'App.py' matches the name of your main Streamlit script
CMD ["streamlit", "run", "App.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
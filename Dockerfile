# Use a lightweight Python base image
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data (vader_lexicon) - this line is crucial for VADER
# Although app.py also has a fallback, it's good practice to have it here too.
RUN python -m nltk.downloader vader_lexicon

# Copy the application code and your dataset into the container
COPY app.py .
COPY Dataset-SA.csv . # Make sure your dataset is in the same directory as app.py

# Expose the port Streamlit runs on
EXPOSE 8501

# Command to run the Streamlit application
# --server.port=8501: Sets the port for Streamlit
# --server.enableCORS=false & --server.enableXsrfProtection=false: Useful for deployment environments
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.enableCORS=false", "--server.enableXsrfProtection=false"]
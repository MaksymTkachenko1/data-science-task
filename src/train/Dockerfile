# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy the local directory contents into the container at /app
COPY . /app

# Install the required Python packages
RUN pip install -r requirements.txt

# Download required NLTK data during the build process to /usr/local/share/nltk_data
RUN mkdir -p /usr/local/share/nltk_data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords wordnet punkt_tab

# Define the command to run the training script
CMD ["python", "src/train/train.py"]
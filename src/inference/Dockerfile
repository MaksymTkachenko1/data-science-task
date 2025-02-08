# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the necessary files into the container
COPY ./src/inference/run_inference.py /app/src/inference/
COPY ./requirements.txt /app/

# Install the required Python packages
RUN pip install -r requirements.txt

# Download required NLTK data during the build process
RUN mkdir -p /usr/local/share/nltk_data
RUN python -m nltk.downloader -d /usr/local/share/nltk_data punkt stopwords wordnet punkt_tab

# Copy the trained model into the container
COPY ./outputs/models/model.pkl /app/outputs/models/

# Define the command to run the inference script
CMD ["python", "src/inference/run_inference.py"]
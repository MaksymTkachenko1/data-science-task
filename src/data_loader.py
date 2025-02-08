import requests
import zipfile
import io
import pandas as pd
import os

def download_and_extract_data(url, data_dir):
    """Downloads a ZIP file and extracts it."""
    response = requests.get(url)
    response.raise_for_status()

    with zipfile.ZipFile(io.BytesIO(response.content)) as zip_file:
        zip_file.extractall(data_dir)

def load_data_to_dataframe(data_dir, file_name):
    """Loads a CSV file into a pandas DataFrame."""
    file_path = os.path.join(data_dir, file_name)
    if os.path.exists(file_path):
        return pd.read_csv(file_path)
    else:
        print(f"Error: File not found at {file_path}")
        return None

def main():
    """Downloads, extracts, and loads the datasets."""
    train_url = 'https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_train_dataset.zip'
    test_url = 'https://static.cdn.epam.com/uploads/583f9e4a37492715074c531dbd5abad2/ds/final_project_test_dataset.zip'

    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, 'data', 'raw')
    train_data_dir = os.path.join(data_dir, 'final_project_train_dataset')
    test_data_dir = os.path.join(data_dir, 'final_project_test_dataset')

    # Create directories
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(train_data_dir, exist_ok=True)
    os.makedirs(test_data_dir, exist_ok=True)

    print("Downloading and extracting training data...")
    download_and_extract_data(train_url, data_dir)

    print("Downloading and extracting test data...")
    download_and_extract_data(test_url, data_dir)

    # Load data into DataFrames
    train_df = load_data_to_dataframe(train_data_dir, 'train.csv')
    test_df = load_data_to_dataframe(test_data_dir, 'test.csv')

    if train_df is not None:
        print("Train data loaded successfully!")
    if test_df is not None:
        print("Test data loaded successfully!")

if __name__ == "__main__":
    main()
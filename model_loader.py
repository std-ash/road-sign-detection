import os
import requests
import torch
from tqdm import tqdm

def download_file(url, destination):
    """
    Download a file from a URL to a destination path with progress bar
    """
    if os.path.exists(destination):
        print(f"File already exists at {destination}")
        return True
    
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()  # Raise an exception for 4XX/5XX responses
        
        # Get file size if available
        file_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(destination), exist_ok=True)
        
        # Show download progress
        print(f"Downloading model from {url} to {destination}")
        progress_bar = tqdm(total=file_size, unit='iB', unit_scale=True)
        
        with open(destination, 'wb') as file:
            for data in response.iter_content(block_size):
                progress_bar.update(len(data))
                file.write(data)
        progress_bar.close()
        
        # Check if download completed successfully
        if file_size != 0 and progress_bar.n != file_size:
            print("ERROR: Download incomplete")
            return False
        
        print(f"Download completed successfully: {destination}")
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def prepare_models():
    """
    Download models if they don't exist
    """
    # Create weights directory if it doesn't exist
    os.makedirs('weights', exist_ok=True)
    
    # Dictionary of models to download: {destination_path: url}
    models = {
        'weights/best.pt': os.environ.get('MODEL_WEIGHTS_URL', ''),
        'weights/yolov5_traffic_signs.pt': os.environ.get('YOLO_WEIGHTS_URL', '')
    }
    
    for dest_path, url in models.items():
        if url and not os.path.exists(dest_path):
            print(f"Downloading model to {dest_path}")
            success = download_file(url, dest_path)
            if not success:
                print(f"Warning: Failed to download model to {dest_path}")
        elif not url and not os.path.exists(dest_path):
            print(f"Warning: No URL provided for {dest_path} and file doesn't exist")

if __name__ == "__main__":
    prepare_models()

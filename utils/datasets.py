import os
import zipfile
from torch.utils.data import DataLoader
import tonic
import tonic.transforms as transforms
import gdown

def download_from_drive(file_id, dest_path):
    """Download dataset zip from Google Drive if it doesn't already exist."""
    if os.path.exists(dest_path):
        print(f"Found existing dataset zip at {dest_path}")
    else:
        print(f"Downloading dataset from Google Drive to {dest_path} ...")
        url = f"https://drive.google.com/uc?id={file_id}"
        gdown.download(url, dest_path, quiet=False)
        print("Download complete.")

def extract_if_needed(zip_path, extract_dir):
    """Extract dataset zip only if not already extracted."""
    marker_file = os.path.join(extract_dir, ".extracted_marker")
    if os.path.exists(marker_file):
        print("Dataset already extracted. Skipping unzip.")
        return

    if not zipfile.is_zipfile(zip_path):
        print(f"{zip_path} is not a valid zip file.")
        return

    print("Extracting dataset ...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_dir)

    with open(marker_file, "w") as f:
        f.write("extracted")
    print(f"Extraction complete. Files ready in {extract_dir}")

def load_dvsgesture(batch_size=32, drive_file_id=None):
    """
    Loads DVS Gesture dataset from local cache or Google Drive.
    If drive_file_id is provided, downloads it automatically.
    """
    save_dir = "./data"
    os.makedirs(save_dir, exist_ok=True)

    if drive_file_id:
        zip_path = os.path.join(save_dir, "dvsgesture.zip")
        download_from_drive(drive_file_id, zip_path)
        extract_if_needed(zip_path, save_dir)

    sensor_size = tonic.datasets.DVSGesture.sensor_size
    transform = transforms.Compose([
        transforms.Denoise(filter_time=10000),
        transforms.ToFrame(sensor_size=sensor_size, n_time_bins=10)
    ])

    print("Loading DVS Gesture dataset ...")
    trainset = tonic.datasets.DVSGesture(save_to=save_dir, train=True, transform=transform)
    testset = tonic.datasets.DVSGesture(save_to=save_dir, train=False, transform=transform)

    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)

    print("Dataset ready.")
    return trainloader, testloader

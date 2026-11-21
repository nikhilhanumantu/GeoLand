"""
Download and extract datasets for GeoLand AI.
Source: Official GitHub Release assets from https://github.com/nikhilhanumantu/GeoLand
"""
import os
import sys
import zipfile
import urllib.request

REPO_RELEASE_URL = "https://github.com/nikhilhanumantu/GeoLand/releases/download/v1.0.0"

DATASETS = {
    "EuroSAT_RGB": {
        "filename": "EuroSAT_RGB.zip",
        "url": f"{REPO_RELEASE_URL}/EuroSAT_RGB.zip",
        "target_dir": "dataset/EuroSAT_RGB",
        "description": "EuroSAT Sentinel-2 RGB Dataset (27,000 patches, 10 classes)"
    },
    "UCMerced_LandUse": {
        "filename": "UCMerced_LandUse.zip",
        "url": f"{REPO_RELEASE_URL}/UCMerced_LandUse.zip",
        "target_dir": "dataset/UCMerced_LandUse",
        "description": "UC Merced Land-Use Aerial Dataset (2,100 high-res aerial images, 21 classes)"
    }
}

def report_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size
    if total_size > 0:
        percent = min(100.0, downloaded * 100.0 / total_size)
        mb_down = downloaded / (1024 * 1024)
        mb_total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {percent:5.1f}% [{mb_down:6.1f} MB / {mb_total:6.1f} MB]")
    else:
        sys.stdout.write(f"\rDownloading: {downloaded / (1024 * 1024):6.1f} MB")
    sys.stdout.flush()

def download_and_extract(dataset_key):
    info = DATASETS[dataset_key]
    dest_dir = "dataset"
    os.makedirs(dest_dir, exist_ok=True)
    
    zip_path = os.path.join(dest_dir, info["filename"])
    target_path = info["target_dir"]

    if os.path.exists(target_path) and os.listdir(target_path):
        print(f"✓ [{dataset_key}] is already downloaded and present in {target_path}")
        return

    print(f"\n=======================================================")
    print(f"Downloading {info['description']}...")
    print(f"URL: {info['url']}")
    print(f"Destination: {zip_path}")
    print(f"=======================================================")
    
    try:
        urllib.request.urlretrieve(info["url"], zip_path, reporthook=report_progress)
        print("\nDownload complete! Extracting archive...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(dest_dir)
        print(f"✓ Extracted successfully to {target_path}")
    except Exception as e:
        print(f"\n❌ Error downloading {dataset_key}: {e}")
        print(f"You can also manually download from {info['url']} and extract into {dest_dir}/")
    finally:
        if os.path.exists(zip_path):
            try:
                os.remove(zip_path)
            except Exception:
                pass

def main():
    print("🛰️ GeoLand AI Dataset Downloader")
    print("================================")
    for key in DATASETS:
        download_and_extract(key)
    print("\n🎉 All datasets ready for training, evaluation, and inference!")

if __name__ == "__main__":
    main()

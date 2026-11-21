import os
import sys
import time
import zipfile
import urllib.request

def download_ucmerced(dest_dir="dataset"):
    os.makedirs(dest_dir, exist_ok=True)
    zip_path = os.path.join(dest_dir, "UCMerced_LandUse.zip")
    extract_to = os.path.join(dest_dir, "UCMerced_LandUse")

    if os.path.exists(extract_to) and len(os.listdir(extract_to)) >= 20:
        print(f"[Download] UC Merced already exists at {extract_to}")
        return extract_to

    url = "https://huggingface.co/datasets/torchgeo/ucmerced/resolve/main/UCMerced_LandUse.zip"
    print(f"[Download] Downloading UC Merced Land Use dataset from Hugging Face...")
    print(f"URL: {url}")
    print(f"Destination: {zip_path}")

    start_time = time.time()
    
    # Custom hook with progress
    def reporthook(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            mb = downloaded / (1024 * 1024)
            tot_mb = total_size / (1024 * 1024)
            speed = mb / max(time.time() - start_time, 0.001)
            sys.stdout.write(f"\rDownloading: {pct:5.1f}% | {mb:6.1f} / {tot_mb:6.1f} MB [{speed:5.1f} MB/s]")
            sys.stdout.flush()

    opener = urllib.request.build_opener()
    opener.addheaders = [('User-Agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    urllib.request.urlretrieve(url, zip_path, reporthook=reporthook)
    print(f"\n[Download] Complete in {time.time() - start_time:.1f}s.")

    print(f"[Extract] Extracting {zip_path} -> {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(dest_dir)

    # Note: UCMerced_LandUse.zip typically extracts to dataset/UCMerced_LandUse/Images/...
    print("[Extract] Extraction completed successfully!")

    # Verify extracted classes
    images_dir = os.path.join(extract_to, "Images") if os.path.exists(os.path.join(extract_to, "Images")) else extract_to
    classes = [d for d in os.listdir(images_dir) if os.path.isdir(os.path.join(images_dir, d))]
    print(f"[Verify] Found {len(classes)} classes in {images_dir}:")
    print(", ".join(classes))

    # Clean up zip file to save disk space
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"[Clean] Removed {zip_path}")

    return images_dir

if __name__ == "__main__":
    download_ucmerced()

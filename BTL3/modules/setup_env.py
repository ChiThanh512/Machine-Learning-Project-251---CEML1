import os
import sys
import subprocess
import zipfile
from pathlib import Path

# ============================================================
# CÀI ĐẶT THƯ VIỆN CẦN THIẾT
# ============================================================
def install_requirements():
    """Cài đặt các thư viện cần thiết cho project."""
    pkgs = [
        "kaggle", "torch", "torchvision", "torchaudio",
        "scikit-learn", "matplotlib", "pandas", "h5py"
    ]
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--upgrade", *pkgs],
        check=True
    )
    print("✅ All required packages are installed and up-to-date.")

# ============================================================
# TẢI DATASET TỪ KAGGLE
# ============================================================
def download_dataset(
    dataset_ref="l3llff/flowers",
    target_dir="/content/dataset",
    kaggle_user="nguyenk512",
    kaggle_key="187454a718c857637f7319f39e33b509"
):
    """Tải và giải nén dataset từ Kaggle."""
    os.environ["KAGGLE_USERNAME"] = kaggle_user
    os.environ["KAGGLE_KEY"] = kaggle_key

    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)

    print(f"📦 Downloading dataset: {dataset_ref} ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", dataset_ref, "-p", str(target), "-q"], check=True)
    print("✅ Download completed.")

    print("🧩 Extracting dataset ...")
    for z in target.glob("*.zip"):
        with zipfile.ZipFile(z, "r") as zip_ref:
            zip_ref.extractall(target)
        z.unlink()
    print("✅ Extraction completed.")

    exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp")
    count = sum(len(list(target.rglob(e))) for e in exts)
    print(f"📸 Found {count} images in {target}")
    return target

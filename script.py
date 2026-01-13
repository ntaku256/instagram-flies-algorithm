import zipfile
import os

zip_path = "2023プレ卒研.zip"
output_dir = "extracted_files"

os.makedirs(output_dir, exist_ok=True)

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    for info in zip_ref.infolist():
        try:
            print(f"Extracting: {info.filename}")
            zip_ref.extract(info, output_dir)
        except Exception as e:
            print(f"❌ Failed to extract {info.filename}: {e}")

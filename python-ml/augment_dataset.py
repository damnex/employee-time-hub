import os
import cv2
import numpy as np
import urllib.request
from pathlib import Path
import csv

DATASET_DIR = Path("dataset")

def get_employee_folders():
    folders = []
    if Path("metadata.generated.csv").exists():
        with open("metadata.generated.csv", "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                folders.append(row["folder_name"])
    if not folders:
        folders = ["emp-1"]
    return folders

def download_face(url, save_path):
    print(f"Downloading {url} to {save_path}")
    req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
    with urllib.request.urlopen(req) as response, open(save_path, 'wb') as out_file:
        data = response.read()
        out_file.write(data)

def augment_image(img_path, output_dir, count=30):
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"Could not read {img_path}")
        return
    
    rows, cols = img.shape[:2]
    
    for i in range(count):
        angle = np.random.uniform(-20, 20)
        M = cv2.getRotationMatrix2D((cols/2, rows/2), angle, 1)
        rotated = cv2.warpAffine(img, M, (cols, rows), borderMode=cv2.BORDER_REPLICATE)
        
        gamma = np.random.uniform(0.6, 1.4)
        invGamma = 1.0 / gamma
        table = np.array([((j / 255.0) ** invGamma) * 255 for j in np.arange(0, 256)]).astype("uint8")
        brightened = cv2.LUT(rotated, table)
        
        if np.random.rand() > 0.5:
            brightened = cv2.GaussianBlur(brightened, (5, 5), 0)
            
        cv2.imwrite(str(output_dir / f"aug_{i}.jpg"), brightened)

def main():
    DATASET_DIR.mkdir(exist_ok=True)
    folders = get_employee_folders()
    
    for i, emp_folder in enumerate(folders):
        emp_dir = DATASET_DIR / emp_folder
        emp_dir.mkdir(exist_ok=True)
        
        base_img_path = emp_dir / "base.jpg"
        if not base_img_path.exists():
            # Use random user generator based on index
            url = f"https://randomuser.me/api/portraits/men/{i+10}.jpg"
            download_face(url, base_img_path)
            
        print(f"Augmenting images for {emp_folder}...")
        augment_image(base_img_path, emp_dir, 30)
        print(f"Done augmenting for {emp_folder}")

if __name__ == "__main__":
    main()

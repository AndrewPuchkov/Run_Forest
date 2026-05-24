import os
import shutil

root_dir = r"D:\run_forest\data\spatial-reasoning"
images_dir = os.path.join(root_dir, "images")
masks_dir = os.path.join(root_dir, "masks")

os.makedirs(images_dir, exist_ok=True)
os.makedirs(masks_dir, exist_ok=True)

for f in os.listdir(root_dir):
    if f.endswith(".tif"):
        if "image" in f:
            shutil.move(os.path.join(root_dir, f), os.path.join(images_dir, f))
        elif "mask" in f:
            shutil.move(os.path.join(root_dir, f), os.path.join(masks_dir, f))

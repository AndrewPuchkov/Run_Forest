import os
import random

root_dir = r"D:\run_forest\data\spatial-reasoning"
root_dir_images = r"D:\run_forest\data\spatial-reasoning\images"
root_dir_masks = r"D:\run_forest\data\spatial-reasoning\masks"
all_files = [f for f in os.listdir(root_dir_images) if f.endswith(".tif")] + [f for f in os.listdir(root_dir_masks) if f.endswith(".tif")]
first_file = all_files[0]
image_files = sorted([f.split('_')[0] for f in all_files if "image" in f])
mask_files = sorted([f.split('_')[0].replace("_image", "_mask") for f in all_files if "image" in f])

assert len(image_files) == len(mask_files), "кол-во не совпадает"

random.shuffle(image_files)

x = len(image_files)
x_train = int(0.7 * x)
x_val = int(0.15 * x)

train_files = image_files[:x_train]
val_files = image_files[x_train:x_train+x_val]
test_files = image_files[x_train+x_val:]

for split, files in zip(["train", "val", "test"], [train_files, val_files, test_files]):
    with open(os.path.join(root_dir, f"{split}_idxs.txt"), "w") as f:
        for file_name in files:
            f.write(file_name + "\n")

print("создано")

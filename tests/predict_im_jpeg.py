import torch
import numpy as np
import os
import rasterio
import cv2
from src.modules import CustomSemanticSegmentationTask
import matplotlib.pyplot as plt

# ===================== НАСТРОЙКИ =====================
MODEL_PATH = r"D:\run_forest\jpg_checkpoints\epoch=29-step=5250.ckpt"
DEVICE = torch.device("cpu")
OUTPUT_DIR = "../predictions"
# ====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Загружаем модель...")
task = CustomSemanticSegmentationTask.load_from_checkpoint(MODEL_PATH, map_location=DEVICE)
model = task.model.eval().to(DEVICE)
print("Модель загружена!\n")


def predict_image(image_path):
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    # === ЧИТАЕМ ИЗОБРАЖЕНИЕ ===
    if image_path.lower().endswith(('.tif', '.tiff')):
        with rasterio.open(image_path) as src:
            img = src.read()  # [C, H, W]
            img = np.transpose(img, (1, 2, 0))  # [H, W, C]
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Загружено: {img.shape} каналов")


    """# === ПРИНУДИТЕЛЬНО ДЕЛАЕМ 4 КАНАЛА ===
    if img.shape[2] == 3:
        print("Изображение 3-канальное → добавляем NIR (зелёный канал)")
        nir = img[:, :, 1]  # зелёный канал как NIR
        img = np.concatenate([img, nir[..., np.newaxis]], axis=2)"""
    if img.shape[0] != 512 or img.shape[1] != 512:
        img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_LINEAR)
        print("→ Изображение ресайзнуто до 512x512")

    # To tensor
    tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    tensor = tensor.unsqueeze(0).to(DEVICE)

    # Предсказание
    with torch.no_grad():
        logits = model(tensor)
        pred_mask = logits.argmax(dim=1).squeeze(0).cpu().numpy()

    # Цветная маска
    colors = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 255]}
    colored = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for c, col in colors.items():
        colored[pred_mask == c] = col

    # Наложение
    overlay = cv2.addWeighted(img[:, :, :3], 0.6, colored, 0.4, 0)

    # Сохранение
    cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_mask.png", colored)
    cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"✅ Готово! Результаты в '{OUTPUT_DIR}'")

    # Показать
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1);
    plt.imshow(img[:, :, :3]);
    plt.title("Оригинал");
    plt.axis("off")
    plt.subplot(1, 3, 2);
    plt.imshow(colored);
    plt.title("Маска");
    plt.axis("off")
    plt.subplot(1, 3, 3);
    plt.imshow(overlay);
    plt.title("Наложение");
    plt.axis("off")
    plt.tight_layout()
    plt.show()


# ===================== ЗАПУСК =====================
if __name__ == "__main__":
    path = r"/Run_Forest/test_segmentation/bloomington27.tif"

    if os.path.exists(path):
        predict_image(path)
    else:
        print("Файл не найден!")
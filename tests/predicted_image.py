import torch
import numpy as np
import os
import rasterio
import cv2  # ← Вот эту строку обязательно добавь!
from src.modules import CustomSemanticSegmentationTask
import matplotlib.pyplot as plt

# ===================== НАСТРОЙКИ =====================
MODEL_PATH = r"D:\run_forest\multiclass\checkpoints\epoch=51-step=136500.ckpt"
DEVICE = torch.device("cpu")
OUTPUT_DIR = "../predictions"
# ====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Загружаем модель...")
task = CustomSemanticSegmentationTask.load_from_checkpoint(MODEL_PATH, map_location=DEVICE)
model = task.model.eval().to(DEVICE)
print("Модель загружена!\n")


def predict_tif(tif_path):
    base_name = os.path.splitext(os.path.basename(tif_path))[0]

    # Читаем tif
    with rasterio.open(tif_path) as src:
        img = src.read()  # [C, H, W]
        img = np.transpose(img, (1, 2, 0))  # [H, W, C]

    print(f"Загружено: {img.shape} (каналов: {img.shape[2]})")

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

    print(f"✅ Готово! Результаты сохранены в папку '{OUTPUT_DIR}'")

    # Показать результат
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(img[:, :, :3])
    plt.title("Оригинал")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(colored)
    plt.title("Маска")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(overlay)
    plt.title("Наложение")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


# ===================== ЗАПУСК =====================
if __name__ == "__main__":
    path = r"/Run_Forest/test_segmentation/412_image_2.tif"

    if os.path.exists(path):
        predict_tif(path)
    else:
        print("Файл не найден!")
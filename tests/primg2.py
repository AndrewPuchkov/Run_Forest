import torch
import numpy as np
import os
import rasterio
import cv2
from src.modules import CustomSemanticSegmentationTask
import matplotlib.pyplot as plt
from pathlib import Path

# ===================== НАСТРОЙКИ =====================
MODEL_PATH = r"D:\run_forest\jpg_checkpoints\epoch=29-step=5250.ckpt"
DEVICE = torch.device("cpu")
OUTPUT_DIR = "../predictions"
TILE_SIZE = 512
OVERLAP = 128  # перекрытие тайлов (чтобы не было швов)
# ====================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("Загружаем модель...")
task = CustomSemanticSegmentationTask.load_from_checkpoint(MODEL_PATH, map_location=DEVICE)
model = task.model.eval().to(DEVICE)
print("Модель загружена!\n")



def predict_large_image(image_path):
    base_name = Path(image_path).stem

    # Читаем изображение
    if image_path.lower().endswith(('.tif', '.tiff')):
        with rasterio.open(image_path) as src:
            img = src.read()
            img = np.transpose(img, (1, 2, 0))  # [H, W, C]
    else:
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"Оригинальный размер: {img.shape}")

    H, W, C = img.shape
    full_mask = np.zeros((H, W), dtype=np.uint8)

    step = TILE_SIZE - OVERLAP

    for y in range(0, H, step):
        for x in range(0, W, step):
            # Вырезаем тайл
            tile = img[y:y + TILE_SIZE, x:x + TILE_SIZE]
            tile_h, tile_w = tile.shape[:2]

            # Дополняем до 512x512, если нужно
            if tile_h != TILE_SIZE or tile_w != TILE_SIZE:
                padded = np.zeros((TILE_SIZE, TILE_SIZE, C), dtype=np.uint8)
                padded[:tile_h, :tile_w] = tile
                tile = padded

            # Предсказание
            tensor = torch.from_numpy(tile).permute(2, 0, 1).float() / 255.0
            tensor = tensor.unsqueeze(0).to(DEVICE)

            with torch.no_grad():
                logits = model(tensor)
                pred = logits.argmax(dim=1).squeeze(0).cpu().numpy()  # 512x512

            # === ИСПРАВЛЕННАЯ ЗАПИСЬ ОБРАТНО ===
            full_mask[y:y + tile_h, x:x + tile_w] = pred[:tile_h, :tile_w]

    # Цветная маска
    colors = {0: [0, 0, 0], 1: [0, 255, 0], 2: [255, 0, 255]}
    colored = np.zeros((H, W, 3), dtype=np.uint8)
    for c, col in colors.items():
        colored[full_mask == c] = col

    # Наложение
    overlay = cv2.addWeighted(img[:, :, :3], 0.6, colored, 0.4, 0)

    # Сохранение
    cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_mask.png", colored)
    cv2.imwrite(f"{OUTPUT_DIR}/{base_name}_overlay.jpg", cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    print(f"✅ Готово! Обработано тайлами {TILE_SIZE}x{TILE_SIZE} с overlap={OVERLAP}")

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
    path = r"/Run_Forest/test_segmentation/bellingham14.tif"
    if os.path.exists(path):
        predict_large_image(path)
    else:
        print("Файл не найден!")
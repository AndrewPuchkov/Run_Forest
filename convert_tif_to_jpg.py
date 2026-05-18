from pathlib import Path
import rasterio
import numpy as np
from PIL import Image
from tqdm import tqdm

# ==================== НАСТРОЙКИ ====================
ROOT = Path(r"D:\run_forest\data\spatial-reasoning")

OUTPUT_ROOT = Path(r"D:\run_forest\data\jpg_data")
OUTPUT_IMAGES = OUTPUT_ROOT / "images"
OUTPUT_MASKS = OUTPUT_ROOT / "masks"

OUTPUT_IMAGES.mkdir(parents=True, exist_ok=True)
OUTPUT_MASKS.mkdir(parents=True, exist_ok=True)

NUM_FILES = 100   # сколько первых файлов взять

print("Получаем список файлов...")

# Правильный способ
image_files = sorted((ROOT / "images").glob("*.tif"))[:NUM_FILES]
mask_files = sorted((ROOT / "masks").glob("*.tif"))[:NUM_FILES]

print(f"Найдено изображений: {len(image_files)}")
print(f"Найдено масок: {len(mask_files)}")

if len(image_files) == 0 or len(mask_files) == 0:
    print("Ошибка: Файлы не найдены! Проверь путь.")
    exit()

# ==================== КОНВЕРТАЦИЯ ====================
print("\nНачинаем конвертацию...")

for i, (tif_img, tif_mask) in enumerate(tqdm(zip(image_files, mask_files))):
    name = tif_img.stem  # например "40730"

    # Конвертируем изображение
    with rasterio.open(tif_img) as src:
        data = src.read()[:3]                    # RGB каналы
        rgb = np.transpose(data, (1, 2, 0))
        Image.fromarray(rgb).save(OUTPUT_IMAGES / f"{name}.jpg", quality=95)

    # Конвертируем маску
    with rasterio.open(tif_mask) as src:
        mask = src.read(1)
        Image.fromarray(mask.astype(np.uint8)).save(OUTPUT_MASKS / f"{name}.png")

print("\n✅ Готово!")
print(f"Изображения → {OUTPUT_IMAGES}")
print(f"Маски → {OUTPUT_MASKS}")
print(f"Сконвертировано: {len(image_files)} пар")
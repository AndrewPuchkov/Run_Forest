import requests
from bs4 import BeautifulSoup
from pathlib import Path
from tqdm import tqdm
import time

url = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/index.html"
save_dir = Path(r"D:\run_forest\mass_roads_test_sat")
save_dir.mkdir(exist_ok=True)

print("Получаем список файлов...")

response = requests.get(url)
soup = BeautifulSoup(response.text, 'html.parser')

links = []
for a in soup.find_all('a'):
    href = a.get('href')
    if href and href.endswith('.tiff'):
        full_url = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/test/sat/" + href
        links.append(full_url)

print(f"Найдено {len(links)} файлов")

for link in tqdm(links):
    filename = Path(link).name
    filepath = save_dir / filename

    if filepath.exists():
        continue

    try:
        r = requests.get(link, stream=True)
        with open(filepath, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        time.sleep(0.3)  # небольшая задержка, чтобы не нагружать сервер
    except Exception as e:
        print(f"Ошибка при скачивании {filename}: {e}")

print("Скачивание завершено!")
print(f"Файлы сохранены в: {save_dir}")
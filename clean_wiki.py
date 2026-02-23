import os
import re

INPUT_DIR = "text"
OUTPUT_FILE = "textKZ.txt"

with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            path = os.path.join(root, file)
            try:
                with open(path, encoding="utf-8") as f:
                    text = f.read()

                text = re.sub(r"<.*?>", "", text)
                text = re.sub(r'\s+', ' ', text)

                sentences = re.split(r'(?<=[.!?])\s+', text.strip())

                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) < 40:
                        continue
                    if not re.search(r"[А-Яа-яӘәІіҢңҒғҮүҰұҚқӨөҺһ]", sentence):
                        continue
                    out.write(sentence + "\n")

            except:
                pass  # пропускаем битые файлы

print("Готово")
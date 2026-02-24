import random
import json

INPUT_FILE = "textKZ.txt"
OUTPUT_FILE = "grammar_dataset.jsonl"

def break_sentence(sentence):
    words = sentence.split()

    if len(words) < 4:
        return None

    mode = random.choice(["shuffle", "remove_suffix"])

    if mode == "shuffle":
        random.shuffle(words)
        return " ".join(words)

    if mode == "remove_suffix":
        broken = []
        for w in words:
            if len(w) > 6 and random.random() < 0.3:
                broken.append(w[:-2])
            else:
                broken.append(w)
        return " ".join(broken)

with open(INPUT_FILE, encoding="utf-8") as f, \
     open(OUTPUT_FILE, "w", encoding="utf-8") as out:

    lines = f.readlines()

    for line in lines[:50000]:
        sentence = line.strip()
        broken = break_sentence(sentence)

        if broken and broken != sentence:
            sample = {
                "instruction": f"Error, fix the grammer: {broken}",
                "response": sentence
            }
            out.write(json.dumps(sample, ensure_ascii=False) + "\n")

print("Done")
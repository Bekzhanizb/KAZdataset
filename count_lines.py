filename = 'textKZ.txt'
count = 0
with open(filename, 'r', encoding='utf-8') as file:
    for line in file:
        count += 1
print(f"Line counts: {count}")

import random

src_path = "./train.list"

with open(src_path, 'r') as f:
    lines = f.readlines()

new_lines = []

for line in lines:
    eles = line.strip().split('\t')
    if len(eles) != 4:
        print(line)
    else:
        new_lines.append('\t'.join(eles[2:]) + '\n')

random.shuffle(new_lines)

train_num = int(len(new_lines) * 0.9)

with open('train.txt', 'w') as f:
    f.writelines(new_lines[:train_num])

with open('valid.txt', 'w') as f:
    f.writelines(new_lines[train_num:])
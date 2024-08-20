import csv
import json

output_list = []

with open('./gather_json/sst2_train.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append([columns[0]])


with open('./gather_json/sst2_dev.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append([columns[0]])

with open('./gather_json/sst2_test.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append([columns[1]])

with open('./gather_json/sst2_all.json', 'w', encoding='utf-8') as file:
    json.dump(output_list, file)
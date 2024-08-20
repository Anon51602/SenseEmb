import csv
import json

output_list = []



with open('./gather_json/cola_train.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append([columns[3]])


with open('./gather_json/cola_train.json', 'w', encoding='utf-8') as file:
    json.dump(output_list, file)


with open('./gather_json/cola_test.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append([columns[1]])


with open('./gather_json/cola_test.json', 'w', encoding='utf-8') as file:
    json.dump(output_list, file)
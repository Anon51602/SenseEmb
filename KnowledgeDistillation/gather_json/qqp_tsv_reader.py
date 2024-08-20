import csv
import json

output_list = []

with open('./gather_json/qqp_train.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[3],columns[4]))


with open('./gather_json/qqp_dev.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[3],columns[4]))

with open('./gather_json/qqp_test.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[1],columns[2]))


with open('./gather_json/qqp_all.json', 'w', encoding='utf-8') as file:
    json.dump(output_list, file)
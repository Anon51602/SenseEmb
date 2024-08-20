import csv
import json

output_list = []


with open('./gather_json/qnli_train.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[2],columns[1]))



with open('./gather_json/qnli_dev.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[2],columns[1]))

with open('./gather_json/qnli_test.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[2],columns[1]))



with open('./gather_json/qnli_all.json', 'w', encoding='utf-8') as file:
    json.dump(output_list, file)
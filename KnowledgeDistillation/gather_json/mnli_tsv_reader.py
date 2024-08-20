import csv
import json

output_list = []


with open('./mnli_dev_matched.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[8],columns[9]))
       

with open('./mnli_dev_mismatched.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[8],columns[9]))


with open('./mnli_test_matched.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[8],columns[9]))
       

with open('./mnli_test_mismatched.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[8],columns[9]))

#with open('./mnli_dev.json', 'w', encoding='utf-8') as file:
#    json.dump(output_list, file)



with open('./mnli_train.tsv', 'r', encoding='utf-8') as file:
    next(file)
    for line in file:
        # Split each line by tab character
        columns = line.strip().split('\t')
        output_list.append((columns[8],columns[9]))




with open('./mnli_all.json', 'w', encoding='utf-8') as file:
    json.dump(output_list, file)
import json
import os

# List of JSON file paths
#file_paths = ["./cola_train.json", "./mnli_train.json","./mrpc_train.json", "./qnli_train.json", "./qqp_train.json", "./rte_train.json", "./sst2_train.json", "./stsb_train.json",
#"./cola_dev.json", "./mnli_dev.json","./mrpc_dev.json", "./qnli_dev.json", "./qqp_dev.json", "./rte_dev.json", "./sst2_dev.json", "./stsb_dev.json"]
file_paths = ["./cola_train.json","./cola_test.json"]
# Initialize an empty list to hold the combined data
combined_list = []

# Loop through each file
for file_path in file_paths:
    with open(file_path, 'r') as file:
        data = json.load(file)
        if isinstance(data, list):
            combined_list.extend(data)

# Write the combined list to a new JSON file
output_path = "cola_tt.json"
with open(output_path, 'w') as output_file:
    json.dump(combined_list, output_file, indent=4)

print(f"Combined JSON saved to {output_path}")

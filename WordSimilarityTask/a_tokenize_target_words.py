import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import pickle
from transformers import BertModel, BertTokenizer


tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


file_path = './target_words.txt'
input_token = []
with open(file_path, 'r') as file:
    for line in file:
        # Strip any leading/trailing whitespace characters, including the newline character
        word = line.strip()
        input_token.append(word)

words_to_tokenid = dict()

for token in input_token:
    token_ids = tokenizer(token, return_tensors="pt", padding=False, add_special_tokens=False)  # .to(model.device)
    token_ids = token_ids['input_ids'][0].tolist()
    print("Token {} ID: {}".format(token, token_ids))
    words_to_tokenid[token] = token_ids


with open("./words_to_tokenid_target_words.pkl", 'wb') as handle:
    pickle.dump(words_to_tokenid, handle)
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer
from transformers import AutoModel, AutoTokenizer
from transformers import AutoTokenizer, DebertaV2Model
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import BertModel, BertTokenizer
import pickle
from tqdm import tqdm

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--txt_file', type=str, default='./wiki.txt',help="input txt file")
parser.add_argument('--output_pkl_file', type=str,default = './wiki.pkl', help="output pickle file")
args = parser.parse_args()

class TrieNode:
    def __init__(self):
        self.children = {}
        self.failure = None
        self.outputs = []

class AhoCorasick:
    def __init__(self):
        self.root = TrieNode()
        self.root.failure = self.root  # Root's failure points to itself

    def add_word(self, word, name):
        node = self.root
        for number in word:
            if number not in node.children:
                node.children[number] = TrieNode()
            node = node.children[number]
        node.outputs.append((name, len(word)))

    def build_failure_links(self):
        from collections import deque
        queue = deque()
        for number, child_node in self.root.children.items():
            child_node.failure = self.root
            queue.append(child_node)

        while queue:
            current_node = queue.popleft()

            for number, child_node in current_node.children.items():
                queue.append(child_node)
                fail_state_node = current_node.failure
                while fail_state_node != self.root and number not in fail_state_node.children:
                    fail_state_node = fail_state_node.failure
                if number in fail_state_node.children:
                    child_node.failure = fail_state_node.children[number]
                else:
                    child_node.failure = self.root
                child_node.outputs += child_node.failure.outputs

    def search(self, text):
        node = self.root
        results = {}
        for position, number in enumerate(text):
            while node != self.root and number not in node.children:
                node = node.failure
            if number in node.children:
                node = node.children[number]
            else:
                continue  # Skip to the next number in text if there's no valid transition

            for name, length in node.outputs:
                start_position = position - length + 1
                if name in results:
                    results[name].append((start_position, position))
                else:
                    results[name] = [(start_position, position)]
        return results

class TextDataset(Dataset):
    def __init__(self, filepath, tokenizer, words_to_tokenid):
        with open(filepath, 'r', encoding='utf-8') as file:
            self.lines = file.readlines()
        self.tokenizer = tokenizer
        self.words_to_tokenid = words_to_tokenid
        self.ac = AhoCorasick()
        for name, pattern in words_to_tokenid.items():
            self.ac.add_word(pattern, name)
        self.ac.build_failure_links()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        line = self.lines[idx].strip()
        # Tokenize the line, adding required model inputs like attention mask
        encoding = self.tokenizer(line, return_tensors='pt', padding='longest', truncation=True)

        token_ids = encoding['input_ids'].squeeze(0).tolist()  # remove batch dimension added by tokenizer
        positions_dict  = self.ac.search(token_ids)

        # Return a dictionary containing both the tokens and additional data
        return {'input_ids': encoding['input_ids'].squeeze(0),  # remove batch dimension added by tokenizer
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'positions_dict': positions_dict}

# Initialize everything and create the DataLoader
tokenizer = BertTokenizer.from_pretrained('bert-base-cased',model_max_length=512)
model = BertModel.from_pretrained('bert-base-cased').to("cuda:0")

model.eval()


def custom_collate_fn(batch):
    # Extract components from batch
    input_ids = [item['input_ids'] for item in batch]
    attention_masks = [item['attention_mask'] for item in batch]
    positions_dicts = [item['positions_dict'] for item in batch]
    # Pad the input_ids and attention_masks
    input_ids_padded = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_masks_padded = pad_sequence(attention_masks, batch_first=True, padding_value=0)
    return {
        'input_ids': input_ids_padded,
        'attention_mask': attention_masks_padded,
        'positions_dicts': positions_dicts
    }


with open('./words_to_tokenid_target_words.pkl', 'rb') as handle:
    words_to_tokenid = pickle.load(handle)

dataset = TextDataset(args.txt_file, tokenizer, words_to_tokenid=words_to_tokenid)
dataloader = DataLoader(dataset, batch_size=10, shuffle=False, collate_fn=custom_collate_fn, num_workers=32)

word_embeddings_dict = {}
with torch.no_grad():
    # Example DataLoader usage:
    for idx, batch in enumerate(tqdm(dataloader)):
        input_ids = batch['input_ids'].to("cuda:0")
        attention_mask = batch['attention_mask'].to("cuda:0")
        positions_dicts = batch['positions_dicts']  # A list of dictionaries

        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = outputs.last_hidden_state.to('cpu')
        batch_size = embeddings.shape[0]

        for i in range(batch_size):
            pos_dict = positions_dicts[i]
            for word, positions in pos_dict.items():
                if word not in word_embeddings_dict:
                    word_embeddings_dict[word] = []

                for start, end in positions:
                    # Aggregate embeddings by averaging over positions
                    if len(word_embeddings_dict[word]) > 2000:
                        continue
                    word_embedding = embeddings[i, start:end+1].mean(dim=0)
                    word_embeddings_dict[word].append(word_embedding)
        if idx % 1000 == 0:
            print(idx)
            with open(args.output_pkl_file, 'wb') as handle:
                pickle.dump(word_embeddings_dict, handle)

for word in word_embeddings_dict:
    word_embeddings_dict[word] = torch.stack(word_embeddings_dict[word])
    print(f"Word: {word}, Embeddings: {word_embeddings_dict[word].shape}")

with open(args.output_pkl_file, 'wb') as handle:
    pickle.dump(word_embeddings_dict, handle)
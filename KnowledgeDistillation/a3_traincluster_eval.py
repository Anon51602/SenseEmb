import torch
from torch.utils.data import Dataset, DataLoader
from torch import nn
from transformers import AutoTokenizer, AutoModel, DebertaV2Model
import numpy as np
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import pickle
import argparse
import einops
import os
from ranger_adabelief import RangerAdaBelief  # Import the Ranger optimizer
from a4_xquantbase import  TeacherModel, StudentModel
from torch.nn.utils.rnn import pad_sequence



def pad2d_withmask(tensor_list):
    # Determine the maximum size of the first dimension across all tensors
    max_size = max(t.shape[0] for t in tensor_list)
    num_tensors = len(tensor_list)
    feature_dim = tensor_list[0].shape[1]

    # Initialize the padded tensor with zeros
    padded_tensor = torch.zeros(num_tensors, max_size, feature_dim)

    # Initialize the mask with zeros (indicating padding)
    mask = torch.zeros(num_tensors, max_size, dtype=torch.bool)
    # Pad each tensor and update the mask
    for i, tensor in enumerate(tensor_list):
        padded_tensor[i, :tensor.shape[0],:tensor.shape[1]] = tensor
        mask[i, :tensor.shape[0]] = 1

    return padded_tensor, mask



def pad1d(tensor_list):
    padded_tensors_tensor = pad_sequence(tensor_list, batch_first=True, padding_value=0)
    return padded_tensors_tensor


def pad2d(tensors, padding_value=0):
    # Determine the max height and width
    max_height = max(tensor.size(0) for tensor in tensors)
    max_width = max(tensor.size(1) for tensor in tensors)

    # Pad each tensor to the same size
    padded_tensors = []
    for tensor in tensors:
        padding = (0, max_width - tensor.size(1), 0, max_height - tensor.size(0))
        padded_tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        padded_tensors.append(padded_tensor)

    # Stack the padded tensors into a single tensor
    padded_tensors_tensor = torch.stack(padded_tensors)

    return padded_tensors_tensor

def pad3d(tensors, padding_value=0):
    # Determine the max depth, height, and width
    max_depth = max(tensor.size(0) for tensor in tensors)
    max_height = max(tensor.size(1) for tensor in tensors)
    max_width = max(tensor.size(2) for tensor in tensors)

    # Pad each tensor to the same size
    padded_tensors = []
    for tensor in tensors:
        padding = (0, max_width - tensor.size(2), 0, max_height - tensor.size(1), 0, max_depth - tensor.size(0))
        padded_tensor = F.pad(tensor, padding, mode='constant', value=padding_value)
        padded_tensors.append(padded_tensor)

    padded_tensors_tensor = torch.stack(padded_tensors)
    return padded_tensors_tensor


def custom_collate_fn(batch):
    transposed = list(zip(*batch))
    padded_tensors = []

    for tensors in transposed:
        if tensors[0].dim() == 1:
            padded_tensors.append(pad1d(tensors))
        elif tensors[0].dim() == 2:
            padded_tensors.append(pad2d(tensors))
        elif tensors[0].dim() == 3:
            padded_tensors.append(pad3d(tensors))
        else:
            raise ValueError("Unsupported tensor dimension: {}".format(tensors[0].dim()))

    return tuple(padded_tensors)

class TokenClassificationDataset(Dataset):
    def __init__(self, tokenizer,json_path,cluster_path):
        self.tokenizer = tokenizer
        with open(json_path, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        path = cluster_path
        with open(path, 'rb') as handle:
            self.token_emb_dict = pickle.load(handle)
        self.omit_token = []

    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        segments = self.data[idx]
        segments = [tokenizer.tokenize(s) for s in segments]
        tokens = ['[CLS]']
        type_ids = [0]
        for i,s in enumerate(segments):
            tokens.extend(s)
            tokens.append('[SEP]')
            type_ids.extend([i]*(len(s)+1))
        tokens = tokens[:512]
        type_ids = type_ids[:512]
        
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1]*len(token_ids)
        pos_ids = list(range(len(token_ids)))
        token_ids = torch.tensor(token_ids)
        input_mask = torch.tensor(input_mask)
        type_ids = torch.tensor(type_ids)
        pos_ids = torch.tensor(pos_ids)

      
        # Tokenizing the sentence
        cluster_mean_list = []
        cluster_filter_list = []
        for i in range(token_ids.shape[0]):
            element = token_ids[i].item()
            if element in self.omit_token or self.token_emb_dict[element] is None  or len(self.token_emb_dict[element]) == 0  :
                cluster_mean_list.append(torch.zeros(1,1024).half())
                cluster_filter_list.append(0)
            else:
                cluster_mean_list.append(self.token_emb_dict[element])
                cluster_filter_list.append(1)

        cluster_mean, cluster_padding_filter = pad2d_withmask(cluster_mean_list)
        cluster_fiter = torch.tensor(cluster_filter_list).bool()
        
        return token_ids,input_mask,type_ids,pos_ids, cluster_mean.half(), cluster_padding_filter.bool(),cluster_fiter



if __name__ == "__main__":
    # Create the parser
    parser = argparse.ArgumentParser(description='Process some integers and floats.')
    parser.add_argument('--epoch', type=int, default=2, help='Epoch')
    parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate')
    parser.add_argument('--ckpt_path', type=str, default="./cola.clpt", help='Student ckpt path')
    parser.add_argument('--load_ckpt_path', type=str, default=None, help='Path for continue student learning')
    parser.add_argument('--model_config', type=str, default="./experiments/glue/config.json", help='Model Configuration for output layer')
    parser.add_argument('--json_path', type=str, default="./gather_json/mrpc_train.json", help='Input dataset path')
    parser.add_argument('--cluster_path', type=str, default='./cola_cluster10.pkl', help='Sense dictionary Path')
    parser.add_argument('--teacher_ckpt_path', type=str, default='./teacher.ckpt', help='Teacher ckpt path')
    parser.add_argument('--num_labels', type=int, default=2, help='Number of Label')
    parser.add_argument('--fp16', type=bool, default=True, help='FP16 or not')
    args = parser.parse_args()
    start_epoch = 0


    # Tokenization and dataset
    tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v3-large", use_fast=False)
    dataset = TokenClassificationDataset( tokenizer, cluster_path=args.cluster_path, json_path=args.json_path)

    # DataLoader
    data_loader = DataLoader(dataset, batch_size=20, shuffle=True, collate_fn=custom_collate_fn)


    teacher_model = TeacherModel.load_model( "deberta-v3-large", args.model_config, num_labels=args.num_labels, drop_out=0.0,cluster_path=None, only_return_hidden=True )
    for param in teacher_model.parameters():
        param.requires_grad = False
    print("Load ckpt")
   
    model_state_dict = torch.load(args.teacher_ckpt_path, map_location=torch.device('cpu'))
    teacher_model.load_state_dict(model_state_dict)
    teacher_model = teacher_model.to("cuda:0")
    if args.fp16:
       teacher_model = teacher_model.half() #fp16


    model = StudentModel.load_model( "deberta-v3-xsmall", args.model_config, num_labels=args.num_labels, drop_out=0.0,cluster_path=None, only_return_hidden=True)
    if args.load_ckpt_path:
        print("Student load model")
        model_state_dict = torch.load(args.load_ckpt_path, map_location=torch.device('cpu'))
        model.load_state_dict(model_state_dict)
        import re

        match = re.search(r'\d+', args.load_ckpt_path)  # Find the first sequence of digits in the filename
        start_epoch = int(match.group()) + 1  # Convert the extracted string to an integer


    copied_parameters = []
    for name_A, param_A in model.named_parameters():
        if name_A in teacher_model.state_dict():
            param_B = teacher_model.state_dict()[name_A]
            if param_A.shape == param_B.shape:
                param_A.data.copy_(param_B.data)
                copied_parameters.append(name_A)

    print("Copied Parameters:")
    for param in copied_parameters:
        print(param)
    if args.fp16:
       model = model.half() #fp16       
    model = model.to("cuda:0")
    loss_fn = nn.CrossEntropyLoss(ignore_index=-1,reduction='mean')

    optimizer = RangerAdaBelief(model.parameters(), lr=args.lr,betas=(0.9,0.999),eps=1e-8,weight_decay=1e-2,weight_decouple=True)

    for epoch in tqdm(range(start_epoch,args.epoch)):
    
        total_correct =0
        total_count = 0
        # Iterating through data in the dataloader
        for batch_idx, (token_ids,input_mask,type_ids,pos_ids, cluster_mean, cluster_padding_filter,cluster_fiter) in enumerate(tqdm(data_loader)):
            input_ids = token_ids.to("cuda:0")
            attention_mask = input_mask.to("cuda:0")
            token_type_ids = type_ids.to("cuda:0")
            position_ids = pos_ids.to("cuda:0")
            cluster_mean = cluster_mean.to("cuda:0")
            cluster_padding_filter = cluster_padding_filter.to("cuda:0")
            cluster_fiter = cluster_fiter.to("cuda:0")
            

            mapped_outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                position_ids=position_ids, output_all_encoded_layers=True)
            with torch.no_grad():
                teacher_outputs = teacher_model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids,
                    position_ids=position_ids, output_all_encoded_layers=True)
            batch_loss = 0
            batch_correct = 0
            batch_count = 0

            mapped_outputs = mapped_outputs.unsqueeze(3)  
            teacher_outputs = teacher_outputs.unsqueeze(3)
    
            map_logits = torch.einsum('bsij,bsjk->bsik',cluster_mean, mapped_outputs )
            map_logits = map_logits[:,:,:,0] * cluster_padding_filter

            teacher_logits = torch.einsum('bsij,bsjk->bsik', cluster_mean, teacher_outputs)
            teacher_logits = teacher_logits[:,:,:,0] * cluster_padding_filter

            # Masking the logits where mask is 0
            map_logits[~cluster_padding_filter] = float('-inf')
            teacher_logits[~cluster_padding_filter] = float('-inf')
            map_labels=torch.argmax(map_logits,dim=2)

            teacher_labels = torch.argmax(teacher_logits,dim=2)
            teacher_labels[~cluster_fiter] = -1

            map_labels_flat = map_labels[cluster_fiter]
            teacher_labels_flat = teacher_labels[cluster_fiter]

            acc = torch.sum(map_labels_flat==teacher_labels_flat)/ map_labels_flat.shape[0]

            map_logits = einops.rearrange(map_logits, ' b s k -> (b s ) k' )
            teacher_labels = einops.rearrange(teacher_labels, ' b s -> (b s ) ' )
       
            # Calculate cross-entropy loss
            loss = loss_fn(map_logits, teacher_labels) 
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            if batch_idx % 50 == 0:
                print("step loss: {:.2f}, step_acc: {:.2f}".format(loss, acc))
            
        torch.save(model.state_dict(), args.ckpt_path[:-5]+str(epoch)+".ckpt")

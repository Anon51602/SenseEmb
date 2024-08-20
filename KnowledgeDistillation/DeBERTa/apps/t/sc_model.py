# Copyright (c) Microsoft, Inc. 2020
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Author: penhe@microsoft.com
# Date: 01/25/2019
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pdb
import torch
from torch.nn import CrossEntropyLoss
import math
import os
import pickle

from ...deberta import *
from ...utils import *

def replace_quant(input_emb,quant_emb,type):
    type = "dot"
    if quant_emb.dim() == 1:
        return quant_emb
    if type == "dot":
        result = torch.matmul(input_emb, quant_emb.T)
        index = torch.argmax(result)
    elif type == "l2":
        diff = quant_emb - input_emb  # Broadcasting subtracts A from each row of B
        squared_diff = diff.pow(2)  # Element-wise squaring
        sum_squared_diff = torch.sum(squared_diff, dim=1)  # Sum across columns
        result = torch.sqrt(sum_squared_diff)
        index = torch.argmin(result)
    output = quant_emb[index, :]
    return output

__all__= ['SCModel']
class SCModel(NNModule):
  def __init__(self, config, num_labels=2, drop_out=None, pre_trained=None):
    super().__init__(config)
    self.num_labels = num_labels
    self._register_load_state_dict_pre_hook(self._pre_load_hook)
    self.deberta = DeBERTa(config, pre_trained=pre_trained)
    if pre_trained is not None:
      self.config = self.deberta.config
    else:
      self.config = config
    pool_config = PoolConfig(self.config)
    output_dim = self.deberta.config.hidden_size
    self.pooler = ContextPooler(pool_config)
    output_dim = self.pooler.output_dim()

    self.classifier = torch.nn.Linear(output_dim, num_labels)
    drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
    self.dropout = StableDropout(drop_out)
    self.apply(self.init_weights)
    self.deberta.apply_state()
    path = './mnli_dev_fp16_cluster.pkl'
    with open(path, 'rb') as handle:
        self.token_emb_dict = pickle.load(handle)
 

  def forward(self, input_ids, type_ids=None, input_mask=None, labels=None, position_ids=None, **kwargs):
    #print(input_ids.shape)
    if input_ids.shape[1] > 512:
       input_ids = input_ids[:,:512]
    outputs = self.deberta(input_ids, attention_mask=input_mask, token_type_ids=type_ids,
        position_ids=position_ids, output_all_encoded_layers=True)
    encoder_layers = outputs['hidden_states']
    encoder_layers = encoder_layers[-1]
    
    #encoder_emb = encoder_layers.to("cuda:0")
    #quant_embeddings = torch.clone(encoder_emb)
    #for i in range(input_ids.shape[0]):
    #  for j in range(1,input_ids.shape[1]):
    #      #total_count+=1
    #      token_id = input_ids[i,j].item()
    #      #if (token_id != 0) and (token_id not in self.omit_token) :
    #      if self.token_emb_dict[token_id] is None:
    #          continue
    #      #quant_count += 1
    #      cluster_emb = self.token_emb_dict[token_id]
    #      temp_emb = replace_quant(encoder_emb[i,j], cluster_emb.to("cuda:0"),type="dot")
    #      quant_embeddings[i,j,:] = temp_emb
    #pooled_output = self.pooler(quant_embeddings.to(encoder_layers.device))
    #print(encoder_layers.shape)
    #print(encoder_layers[1:].shape)
    pooled_output = self.pooler(encoder_layers[:,1:,:])
    pooled_output = self.dropout(pooled_output)
    logits = self.classifier(pooled_output)
    loss = torch.tensor(0).to(logits)
    if labels is not None:
      if self.num_labels ==1:
        # regression task
        loss_fn = torch.nn.MSELoss()
        logits=logits.view(-1).to(labels.dtype)
        loss = loss_fn(logits, labels.view(-1))
      elif labels.dim()==1 or labels.size(-1)==1:
        label_index = (labels >= 0).nonzero()
        labels = labels.long()
        if label_index.size(0) > 0:
          labeled_logits = torch.gather(logits, 0, label_index.expand(label_index.size(0), logits.size(1)))
          labels = torch.gather(labels, 0, label_index.view(-1))
          loss_fct = CrossEntropyLoss()
          loss = loss_fct(labeled_logits.view(-1, self.num_labels).float(), labels.view(-1))
        else:
          loss = torch.tensor(0).to(logits)
      else:
        log_softmax = torch.nn.LogSoftmax(-1)
        label_confidence = 1
        loss = -((log_softmax(logits)*labels).sum(-1)*label_confidence).mean()
    return {
            'logits' : logits,
            'loss' : loss
          }

  def export_onnx(self, onnx_path, input):
    del input[0]['labels'] #= input[0]['labels'].unsqueeze(1)
    torch.onnx.export(self, input, onnx_path, opset_version=13, do_constant_folding=False, \
        input_names=['input_ids', 'type_ids', 'input_mask', 'position_ids', 'labels'], output_names=['logits', 'loss'], \
        dynamic_axes={'input_ids' : {0 : 'batch_size', 1: 'sequence_length'}, \
          'type_ids' : {0 : 'batch_size', 1: 'sequence_length'}, \
          'input_mask' : {0 : 'batch_size', 1: 'sequence_length'}, \
          'position_ids' : {0 : 'batch_size', 1: 'sequence_length'}, \
     #     'labels' : {0 : 'batch_size', 1: 'sequence_length'}, \
          })

  def _pre_load_hook(self, state_dict, prefix, local_metadata, strict,
      missing_keys, unexpected_keys, error_msgs):
    new_state = dict()
    bert_prefix = prefix + 'bert.'
    deberta_prefix = prefix + 'deberta.'
    for k in list(state_dict.keys()):
      if k.startswith(bert_prefix):
        nk = deberta_prefix + k[len(bert_prefix):]
        value = state_dict[k]
        del state_dict[k]
        state_dict[nk] = value

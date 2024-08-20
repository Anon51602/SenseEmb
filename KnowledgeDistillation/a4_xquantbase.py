
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

from DeBERTa.deberta import *
from DeBERTa.utils import *

import torch
from torch import nn
import torch.nn.functional as F
import pickle
from transformers import AutoTokenizer, AutoModel
from DeBERTa.deberta.config import ModelConfig
from DeBERTa.deberta.cache_utils import load_model_state
import copy


__all__= ['StudentModel']
class StudentModel(NNModule):
  def __init__(self, config, num_labels=2, drop_out=None, pre_trained=None,cluster_path=None, only_return_hidden=False,three_layers=False):
    super().__init__(config)
    self.num_labels = num_labels
    self._register_load_state_dict_pre_hook(self._pre_load_hook)
    self.deberta = DeBERTa(config, pre_trained=pre_trained)
    if pre_trained is not None:
      self.config = self.deberta.config
    else:
      self.config = config
    self.only_return_hidden = only_return_hidden
    self.dim_mapper = nn.Linear(384, 1024)  

    
    pool_config = PoolConfig(self.config)
    hidden_dim = 1024
    self.pooler = ContextPooler(pool_config,hidden_dim)
    output_dim = self.pooler.output_dim()
    self.classifier = torch.nn.Linear(output_dim, num_labels)
    drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
    self.dropout = StableDropout(drop_out)
    self.apply(self.init_weights)
    self.deberta.apply_state()

  def forward(self, input_ids, type_ids=None, input_mask=None, labels=None, position_ids=None, **kwargs):

    outputs = self.deberta(input_ids, attention_mask=input_mask, token_type_ids=type_ids,
        position_ids=position_ids, output_all_encoded_layers=True)
    encoder_layers = outputs['hidden_states'][-1]
    encoder_embeddings = encoder_layers.to("cuda:0")
    encoder_embeddings = self.dim_mapper(encoder_embeddings)

    if self.only_return_hidden:
      return encoder_embeddings
 
    mapped_outputs = torch.clone(encoder_embeddings)
    mapped_outputs = mapped_outputs.unsqueeze(3)  
    cluster_mean = kwargs['cluster_mean']
    cluster_padding_filter = kwargs['cluster_padding_filter']
    map_logits = torch.einsum('bsij,bsjk->bsik',cluster_mean, mapped_outputs )
    map_logits = map_logits[:,:,:,0] * cluster_padding_filter
    map_logits[~cluster_padding_filter] = float('-inf')
    map_labels=torch.argmax(map_logits,dim=2)
    labels_expanded = map_labels.unsqueeze(-1).expand(-1, -1, 1024)
    sense_emb = torch.gather(cluster_mean, 2, labels_expanded.unsqueeze(2)).squeeze(2)
    sense_emb[input_mask[0],input_mask[1],:] = encoder_embeddings[input_mask[0],input_mask[1],:] 
 
    # output layer
    pooled_output = self.pooler(sense_emb[:,1:,:].to(encoder_layers.device))
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
             'emb' : encoder_layers,
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



__all__= ['TeacherModel']
class TeacherModel(NNModule):
  def __init__(self, config, num_labels=2, drop_out=None, pre_trained=None,cluster_path=None, only_return_hidden=False):
    super().__init__(config)
    self.num_labels = num_labels
    print("Num Labels:", num_labels)
    self._register_load_state_dict_pre_hook(self._pre_load_hook)
    self.deberta = DeBERTa(config, pre_trained=pre_trained)
    if pre_trained is not None:
      self.config = self.deberta.config
    else:
      self.config = config
    self.only_return_hidden = only_return_hidden
    pool_config = PoolConfig(self.config)
    output_dim = self.deberta.config.hidden_size
    self.pooler = ContextPooler(pool_config)
    output_dim = self.pooler.output_dim()
   
    self.classifier = torch.nn.Linear(output_dim, num_labels)
    drop_out = self.config.hidden_dropout_prob if drop_out is None else drop_out
    self.dropout = StableDropout(drop_out)
    self.apply(self.init_weights)
    self.deberta.apply_state()
    self.cluster_path = cluster_path


  def forward(self, input_ids, type_ids=None, input_mask=None, labels=None, position_ids=None, **kwargs):
    outputs = self.deberta(input_ids, attention_mask=input_mask, token_type_ids=type_ids,
        position_ids=position_ids, output_all_encoded_layers=True)
    encoder_layers = outputs['hidden_states'][-1]
    encoder_embeddings = encoder_layers.to("cuda:0")
    if self.only_return_hidden:
      return  encoder_layers
    
    mapped_outputs = torch.clone(encoder_embeddings)
    mapped_outputs = mapped_outputs.unsqueeze(3)  
    cluster_mean = kwargs['cluster_mean']
    cluster_padding_filter = kwargs['cluster_padding_filter']
    map_logits = torch.einsum('bsij,bsjk->bsik',cluster_mean, mapped_outputs )
    map_logits = map_logits[:,:,:,0] * cluster_padding_filter
    map_logits[~cluster_padding_filter] = float('-inf')
    map_labels=torch.argmax(map_logits,dim=2)
    labels_expanded = map_labels.unsqueeze(-1).expand(-1, -1, 1024)
    sense_emb = torch.gather(cluster_mean, 2, labels_expanded.unsqueeze(2)).squeeze(2)
    sense_emb[input_mask[0],input_mask[1],:] = encoder_embeddings[input_mask[0],input_mask[1],:] 
 

    if self.cluster_path is not None:
      mapped_outputs = torch.clone(encoder_embeddings)
      mapped_outputs = mapped_outputs.unsqueeze(3)  
      cluster_mean = kwargs['cluster_mean']
      cluster_padding_filter = kwargs['cluster_padding_filter']
      map_logits = torch.einsum('bsij,bsjk->bsik',cluster_mean, mapped_outputs )
      map_logits = map_logits[:,:,:,0] * cluster_padding_filter
      map_logits[~cluster_padding_filter] = float('-inf')
      map_labels=torch.argmax(map_logits,dim=2)
      labels_expanded = map_labels.unsqueeze(-1).expand(-1, -1, 1024)
      sense_emb = torch.gather(cluster_mean, 2, labels_expanded.unsqueeze(2)).squeeze(2)
      sense_emb[input_mask[0],input_mask[1],:] = encoder_embeddings[input_mask[0],input_mask[1],:] 
      pooled_output = self.pooler(sense_emb[:,1:,:].to(encoder_layers.device))
    else:
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
             'emb' : encoder_layers,
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

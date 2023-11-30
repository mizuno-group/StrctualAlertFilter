# -*- coding: utf-8 -*-
"""
Created on Thu 24 15:45 2022

model architecture

@author: NemotoS
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from transformers.modeling_utils import Conv1D
from transformers.models.gpt2.modeling_gpt2 import *
from transformers.models.gpt2.configuration_gpt2 import GPT2Config

import numpy as np
import math

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class Config(GPT2Config):
    def __init__(
        self,
        vocab_size = 78,
        n_positions = 250,
        n_embd = 512,
        n_layer = 8,
        n_head = 8,
        dropout = 0,
        layer_norm_epsilon = 1e-5,
        initializer_range = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range

class Config2(GPT2Config):
    def __init__(
        self,
        vocab_size = 43,
        n_positions = 150,
        n_embd = 256,
        n_layer = 8,
        n_head = 8,
        dropout = 0,
        layer_norm_epsilon = 1e-5,
        initializer_range = 0.02,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.n_positions = n_positions
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.n_head = n_head
        self.dropout = dropout
        self.layer_norm_epsilon = layer_norm_epsilon
        self.initializer_range = initializer_range


class PositionalEncoding(nn.Module):
    def __init__(self,n_embd,dropout,max_len=250):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len,n_embd)
        position = torch.arange(0,max_len).unsqueeze(1) #[maxlen,1]
        div_term = torch.exp(torch.arange(0,n_embd,2) * 
                             -(math.log(10000.0) / n_embd))
        pe[:,0::2] = torch.sin(position*div_term)
        pe[:,1::2] = torch.cos(position*div_term)
        pe = pe.unsqueeze(1) # [maxlen,1,d]
        self.register_buffer("pe",pe)

    def forward(self,x):
        # x: [L,B,D]
        x = x + Variable(self.pe[:x.size(0)],
                         requires_grad=False)
        return self.dropout(x)


class Attention(GPT2Attention):
    def __init__(self,n_ctx,config,scale=False):
        super().__init__(config)
        nx = config.n_embd
        self.n_head = config.n_head
        self.split_size = nx
        self.scale = scale
        self.head_dim = nx // config.n_head

        self.c_attn = Conv1D(3*nx,nx)
        self.c_proj = Conv1D(nx,nx)
        self.attn_dropout = nn.Dropout(config.dropout)

    def _attn(self,q,k,v,attention_mask=None):
        w = torch.matmul(q,k) # [B,H,L,L]
        w = w / math.sqrt(v.size(-1))
        if attention_mask is not None:
            w = w + attention_mask

        w = nn.Softmax(dim=-1)(w)
        w = self.attn_dropout(w)
        outputs = torch.matmul(w,v) # [B,H,L,D/H]
        return outputs

    def forward(self,x,attention_mask=None,layer_past=None):
        """
        x: [L,B,D]
        """
        x = self.c_attn(x).transpose(0,1) # [B,L,3D]
        query, key, value = x.split(self.split_size,dim=2) # [B,L,D] * 3
        query = self._split_heads(query,self.n_head,self.head_dim) # [B,H,L,D/H]
        key = self._split_heads(key,self.n_head,self.head_dim).transpose(-2,-1) # [B,H,D/H,L]
        value = self._split_heads(value,self.n_head,self.head_dim)
        if layer_past is not None:
            past_key, past_value = layer_past[0].transpose(-2,-1), layer_past[1]
            key = torch.cat((past_key,key),dim=-1)
            value = torch.cat((past_value,value),dim=-2)
        present = torch.stack((key.transpose(-2,-1),value)) # [B,L,2D]

        a = self._attn(query,key,value,attention_mask) # [B,H,L,D/H]
        a = self.attn_dropout(self.c_proj(self._merge_heads(a,self.n_head,self.head_dim))) # [B,L,D]
        outputs = [a.transpose(0,1),present]
        return outputs # [L,B,D]


class TransformerBlock(nn.Module):
    def __init__(self,n_ctx,config,scale=False):
        super().__init__()
        nx = config.n_embd
        self.ln_1 = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.attn = Attention(n_ctx,config,scale)
        self.ln_2 = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.mlp = GPT2MLP(4*nx,config)

    def forward(self,x,attention_mask=None,layer_past=None):
        # x: [L,B,D]
        output_attn = self.attn(self.ln_1(x),layer_past=layer_past,attention_mask=attention_mask)
        a = output_attn[0]
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        outputs = [x] + output_attn[1:]
        return outputs


class Encoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        nx = config.n_embd
        self.wte = nn.Embedding(config.vocab_size,nx)
        self.wpe = PositionalEncoding(nx,config.dropout,max_len=config.n_positions)
        self.drop = nn.Dropout(config.dropout)
        
        self.h = nn.ModuleList([TransformerBlock(config.n_positions,config,scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.ln_mem1 = nn.LayerNorm(nx)
        self.ln_mem2 = nn.LayerNorm(nx)
        self.ln_mem3 = nn.LayerNorm(nx)
        self.fc_latent = nn.Linear(3*nx,config.n_embd)

    def create_enc_attention_mask(self,input_ids):
        l, b = input_ids.size()
        pad_array = (input_ids == 0).transpose(0,1).unsqueeze(1).unsqueeze(2)
        return torch.where(pad_array == True, float("-inf"), 0.0) # [B,1,1,L]

    def memory_pool(self,memory):
        mx = torch.max(memory,dim=0)[0]
        ave = torch.mean(memory,dim=0)
        first = memory[0]
        return torch.cat([self.ln_mem1(mx),self.ln_mem2(ave),self.ln_mem3(first)],dim=1)

    def forward(self,x,past=None):
        """
        x: Tensor, [L,B]
        """ 
        input_shape = x.size()
        x = x.view(-1,input_shape[-1])
        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        input_embeds = self.wte(x)
        hidden_states = self.wpe(input_embeds)

        attention_mask = self.create_enc_attention_mask(x)
        output_shape = input_shape + (hidden_states.size(-1),)

        for i, (block,layer_past) in enumerate(zip(self.h,past)):
            outputs = block(hidden_states,layer_past=layer_past,attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
        hidden_states = self.ln_f(hidden_states)
        hidden_states = hidden_states.view(*output_shape) # [L,B,D]

        latent_space = self.memory_pool(hidden_states) # [B,3*D]
        latent_space = self.fc_latent(latent_space)
        return torch.tanh(latent_space) # [B,D]       


class Decoder(nn.Module):
    def __init__(self,config):
        super().__init__()
        nx = config.n_embd
        self.wte = nn.Embedding(config.vocab_size,nx)
        self.wpe = PositionalEncoding(nx,config.dropout,max_len=config.n_positions)
        self.input_proj = nn.Linear(nx,nx,bias=False)
        self.h = nn.ModuleList([TransformerBlock(config.n_positions,config,scale=True) for _ in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(nx,eps=config.layer_norm_epsilon)
        self.output_fc = nn.Linear(nx,config.vocab_size)

    def create_dec_attention_mask(self,input_ids):
        l, b = input_ids.size()
        pad_array = (input_ids == 0).transpose(0,1).unsqueeze(1).unsqueeze(2) # [B,1,1,L]

        seq_array = torch.triu(torch.full((l,l),True,device=DEVICE),diagonal=1)
        seq_array = seq_array.unsqueeze(0).unsqueeze(1) # [1,1,L,L]
        res = torch.logical_or(pad_array,seq_array)
        return torch.where(res == True, float("-inf"), 0.0) # [B,1,L,L]

    def forward(self,x,latent,layer_past=None):
        # x: [L,B]
        # latent: [B,D]
        input_shape = x.size()
        if layer_past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)
        
        attention_mask = self.create_dec_attention_mask(x)
        input_embeds = self.wte(x)
        hidden_states = self.wpe(input_embeds)
        hidden_states = hidden_states + latent.unsqueeze(1).transpose(0,1)

        presents = ()
        for i,(block,layer_past) in enumerate(zip(self.h,past)):
            outputs = block(hidden_states,layer_past=layer_past,attention_mask=attention_mask)
            hidden_states, present = outputs[:2]
            presents = presents + (present,)
        hidden_states = self.ln_f(hidden_states)
        hidden_states = self.output_fc(hidden_states)
        return hidden_states # [L,B,V]


class TransformerModel(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config = config
        self.encoder = Encoder(config)
        self.decoder = Decoder(config)

    def forward(self,src,tgt,past=None):
        latent_space = self.encoder(src)
        outputs = self.decoder(tgt,latent_space,layer_past=past)
        return outputs, latent_space


def warmup_schedule(warmup):
    def f(e):
        if e > 0:
            return min(e**-0.5,e*(warmup**-1.5))
        else:
            return 0
    return f


class EarlyStopping():
    def __init__(self,mode="min",min_delta=0,patience=10,percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode,min_delta,percentage)

        if patience == 0:
            self.is_better = lambda a,b: True
            self.step = lambda a: False

    def step(self,metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics,self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            print("terminating because of early stopping.")
            return True
        
        return False

    def _init_is_better(self,mode,min_delta,percentage):
        if mode not in {"min","max"}:
            raise ValueError("mode "+mode+" is unknown!")
        if not percentage:
            if mode == "min":
                self.is_better = lambda a, best: a < best - min_delta
            if mode == "max":
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == "min":
                self.is_better = lambda a, best: a < best - (best*min_delta/100)
            if mode == "max":
                self.is_better = lambda a, best: a > best + (best*min_delta/100)

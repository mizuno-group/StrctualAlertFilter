# -*- coding: utf-8 -*-
"""
Created on 11/11/2023

prepare dataloader

@author: Y.Kikuchi
"""

import numpy as np
import pandas as pd
import re
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence

CHARSET = {"smiles":r"Cl|Br|[#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps\/\\]",
           "smarts":r'Cl|Br|Hg|Fe|As|Sb|Zn|Se|se|Te|Si|Mg|[!#$&\(\)\*\+\,\-\./0123456789:;=@ABCDFHIMNOPRSTXZ\[\\\]\abceghilnorsv~]'}
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def rank_of_elements(numbers,batch_size):
        # 各要素とそのインデックスの組を作成
        elements_with_index = [(value, index) for index, value in enumerate(numbers)]

        # 要素でソート
        elements_with_index.sort(key=lambda x: x[0])

        # 元のインデックスを取得して結果をリストに格納
        ranks = [index for value, index in elements_with_index]

        result = [rank // batch_size for rank in ranks]

        return result

class BucketSampler(Sampler):
    def __init__(self,dataset,buckets=(20,110,10),shuffle=True,batch_size=128,drop_last=False,device="cuda",Umap=False):
        super().__init__(dataset)
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.drop_last = drop_last
        length = []
        
        for (src, tgt), index in dataset: 
            length.append(len(src))

        assert isinstance(buckets,tuple)
        bmin, bmax, bstep = buckets
        assert (bmax - bmin) % bstep == 0

        num = len(dataset) // batch_size
        #buc = torch.linspace(batch_size,batch_size,num+1)
        bucket_range = np.arange(*buckets)
        #buc = torch.bucketize(torch.tensor(length),torch.tensor(bucket_range),right=False)
        buc = torch.tensor(rank_of_elements(length,batch_size))

    
        bucs = defaultdict(list)
        bucket_max = max(np.array(buc))
        for i,v in enumerate(buc):
            bucs[v.item()].append(i)
        #if Umap == False:
        #    _ = bucs.pop(bucket_max)
        
        self.buckets = dict()
        for bucket_size, bucket in bucs.items():
            if len(bucket) > 0:
                self.buckets[bucket_size] = torch.tensor(bucket,dtype=torch.int,device=device)
        self.__iter__()

    def __iter__(self):
        if self.shuffle == True:
            for bucket_size in self.buckets.keys():
                self.buckets[bucket_size] = self.buckets[bucket_size][torch.randperm(self.buckets[bucket_size].nelement())]

        batches = []
        for bucket in self.buckets.values():
            curr_bucket = torch.split(bucket,self.batch_size)
            if len(curr_bucket) > 1 and self.drop_last == True:
                if len(curr_bucket[-1]) < len(curr_bucket[-2]):
                    curr_bucket = curr_bucket[:-1]
            batches += curr_bucket

        self.length = len(batches)
        if self.shuffle == True:
            random.shuffle(batches)
        return iter(batches)

    def __len__(self):
        return self.length

class Seq2id_Dataset(Dataset):
    def __init__(self,x,y,charset):
        self.input = seq2id(x,vocab_dict(charset),charset)
        self.output = seq2id(y,vocab_dict(charset),charset)
        self.datanum = len(x)

    def __len__(self):
        return self.datanum

    def __getitem__(self,idx):
        out_i = self.input[idx]
        out_o = self.output[idx]
        #index = torch.tensor([idx])
        index = idx
        #category = self.train_df["category"].to_list()[index]
        return (out_i, out_o) ,index

class SFL_Dataset(Seq2id_Dataset):
    def __init__(self,x,y):
        self.tokens = tokens_table()
        self.input = sfl_seq2id(x,self.tokens)
        self.output = sfl_seq2id(y,self.tokens)
        self.datanum = len(x)


def prep_loader(data_x,data_y,buckets=(20,110,10),batch_size=128,shuffle=True,drop_last=True,device=DEVICE,charset="smarts",Umap=False):
    datasets = Seq2id_Dataset(data_x,data_y,charset)
    bucket_sampler = BucketSampler(datasets,buckets=buckets,shuffle=shuffle,batch_size=batch_size,
                                   drop_last=drop_last,device=device,Umap=False)

    train_loader = DataLoader(datasets,batch_sampler=bucket_sampler,collate_fn=collate)            
    return train_loader

def prep_loader_sfl(data_x,data_y,buckets=(20,110,10),batch_size=128,shuffle=True,drop_last=False,device=DEVICE):
    datasets = SFL_Dataset(data_x,data_y)
    bucket_sampler = BucketSampler(datasets,buckets=buckets,shuffle=shuffle,batch_size=batch_size,
                                   drop_last=drop_last,device=device)
    train_loader = DataLoader(datasets,batch_sampler=bucket_sampler,collate_fn=collate)
    return train_loader


def vocab_dict(charset="smiles"):
    regex_sml = CHARSET[charset]
    a = r"Cl|Br|#%\)\(\+\-1032547698:=@CBFIHONPS\[\]cionps\/\\"
    b = r'Cl|Br|Hg|Fe|As|Sb|Zn|Se|se|Te|Si|Mg|[!#$&\(\)\*\+\,\-\./0123456789:;=@ABCDFHIMNOPRSTXZ\[\\\]\abceghilnorsv~]'
    if charset == "smiles":
        temp = re.findall(regex_sml,a)
    elif charset == "smarts":
        temp = re.findall(regex_sml,b)
    temp = sorted(set(temp),key=temp.index)
    vocab_smi = {}
    for i,v in enumerate(temp):
        vocab_smi[v] = i+3
    vocab_smi.update({"<pad>":0,"<s>":1,"</s>":2})
    return vocab_smi

def seq2id(seq_list,vocab,charset="smiles"):
    regex_sml = CHARSET[charset]
    idx_list = []
    ap = idx_list.append
    for v in seq_list:
        char = re.findall(regex_sml,v)
        seq = np.array([vocab[w] for w in char])
        seq = np.concatenate([np.array([1]),seq,np.array([2])]).astype(np.int32)
        ap(seq)
    return idx_list


def collate(batch):
    xs, ys, idx = [], [], []
    for (x,y), index in batch:
        xs.append(torch.LongTensor(x))
        ys.append(torch.LongTensor(y))
        idx.append(index)
    xs = pad_sequence(xs,batch_first=False,padding_value=0)
    ys = pad_sequence(ys,batch_first=False,padding_value=0)
    return (xs, ys), idx


class tokens_table():
    def __init__(self):
        tokens = ['<pad>','<s>','</s>','0','1','2','3','4','5','6','7','8','9','(',')','=','#','@','*','%',
                  '.','/','\\','+','-','c','n','o','s','p','H','B','C','N','O','P','S','F','L','R','I',
                  '[C@H]','[C@@H]','[C@@]','[C@]','[CH2-]','[CH-]','[C+]','[C-]','[CH]','[C]','[H+]','[H]',
                  '[n+]','[nH]','[N+]','[NH+]','[NH-]','[N+]','[N@]','[N@@]','[NH2+]','[N-]','[N]''[NH]',
                  '[O+]','[O-]','[OH-]','[O]','[S]','[S+]','[s+]','[S@]','[S@@]','[B-]','[P]','[P+]','[P@]','[P@@]',
                  '[Cl]','[Cl-]','[I-]','[Br-]',"[Si]"]
        self.table = tokens
        self.id2sm = {i:v for i,v in enumerate(tokens)}
        self.dict = {w:v for v,w in self.id2sm.items()}
        self.table_len = len(self.table)
    
class smarts_tokens_table():
    def __init__(self):
        tokens = ['<pad>','<s>','</s>','0','1','2','3','4','5','6','7','8','9','(',')','=','#','@','*','%',
                  '.','/','\\','+','-','c','n','o','s','p','H','B','C','N','O','P','S','F','L','R','I',
                  '[C@H]','[C@@H]','[C@@]','[C@]','[CH2-]','[CH-]','[C+]','[C-]','[CH]','[C]','[H+]','[H]',
                  '[n+]','[nH]','[N+]','[NH+]','[NH-]','[N+]','[N@]','[N@@]','[NH2+]','[N-]','[N]''[NH]',
                  '[O+]','[O-]','[OH-]','[O]','[S]','[S+]','[s+]','[S@]','[S@@]','[B-]','[P]','[P+]','[P@]','[P@@]',
                  '[Cl]','[Cl-]','[I-]','[Br-]',"[Si]"]
        self.table = tokens
        self.id2sm = {i:v for i,v in enumerate(tokens)}
        self.dict = {w:v for v,w in self.id2sm.items()}
        self.table_len = len(self.table)


def sfl_tokenize(smiles,token_list):
    tokenized = []
    for smile in smiles:
        smile = smile.replace("Br","R").replace("Cl","L")
        char = ""
        tok = []
        for s in smile:
            char += s
            if char in token_list:
                tok.append(char)
                char = ""
        tokenized.append(tok)
    return tokenized

def one_hot_encoder(tokenized,token_dict):
    encoded = []
    for token in tokenized:
        enc = np.array([token_dict[v] for v in token])
        enc = np.concatenate([np.array([1]),enc,np.array([2])]).astype(np.int32)
        encoded.append(enc)
    return encoded

def sfl_seq2id(smiles,tokens):
    tokenized = sfl_tokenize(smiles,tokens.table)
    encoded = one_hot_encoder(tokenized,tokens.dict)
    return encoded

import torch
import sentencepiece as spm
import numpy as np

class Data:
    def __init__(self, text, device='cpu'):
        self.tokenizer = Tokenizer()
        self.device = device
        # vocab = sorted(list(set(text))) # 词表
        train_ls = self.encode_data(text['train'])
        val_ls = self.encode_data(text['val'])
        self.data = {
            'train': torch.tensor(train_ls,dtype=torch.long,device=device),
            'val': torch.tensor(val_ls,dtype=torch.long,device=device)
        }
    
    def get_batch(self, data_set, batch_size, max_len):
        batch_data = self.data['train'] if data_set == "train" else self.data['val']
        assert len(batch_data) > max_len + 1
        
        # 得到随机数，代表训练数据中随机一个起始位置
        ix = torch.randint(len(batch_data) - max_len,(batch_size,),device=self.device) 
        
        # 堆叠成一个(batch_size,max_len)的张量
        # x = torch.stack([batch_data[i:i + max_len] for i in ix])
        # y = torch.stack([batch_data[i + 1:i + max_len + 1] for i in ix])
        
        ix_col = ix.unsqueeze(1) # (batch_size,1)
        offset = torch.arange(max_len,device=self.device).unsqueeze(0) # (1,max_len)
        x = batch_data[ix_col + offset] # (batch_size,max_len)
        y = batch_data[ix_col + offset + 1]
        return x,y
    
    def encode_data(self,poems):
        ids = []
        for poem in poems:
            ids.extend([self.tokenizer.bos_id] + self.tokenizer.encode(poem) + [self.tokenizer.eos_id])
        return ids

class Tokenizer:
    def __init__(self):
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load('./data/input_poems.model')
        self.vocab_size = self.sp.vocab_size()
        self.bos_id = self.sp.bos_id()
        self.eos_id = self.sp.eos_id()
        # self.stoi = {ch : i for i,ch in enumerate(vocab)} # 字符 -> 索引
        # self.itos = {i : ch for i,ch in enumerate(vocab)} # 索引 -> 字符

    
    def encode(self,text):
        return self.sp.EncodeAsIds(text)
    
    def decode(self,ids): # 解码
        return self.sp.DecodeIds(ids)
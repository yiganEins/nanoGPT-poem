import torch
from torch.nn import functional as F
from data.get_data import Data
from nanoGPT.model import GPT
import sentencepiece as spm
import pandas as pd
import math
import random
import os
import json


torch.manual_seed(1337) # 随机数种子
random.seed(1337)

save_dir = './log/test_3'

os.makedirs(save_dir, exist_ok=True)
ckpt_path = f'{save_dir}/GPT.pt'
loss_path = f'{save_dir}/loss.tsv'
config_path = f'{save_dir}/config.json'

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('device: ',device)

iters = 8000
learning_rate = 0.0006
min_lr = learning_rate * 0.08
batch_size = 64
max_len = 256

d_model = 256
nums_layer = 10
nums_head = 8
dropout = 0.2

temperature = 0.7
top_k = 50

eval_iters = 50
eval_interval = 200
warmup_iters = int(iters * 0.05)

# 早停参数
early_stop_patience = 10
early_stop_min_delta = 0.001

cfg = {
    'iters': iters,
    'learning_rate': round(learning_rate,8),
    'min_lr': round(min_lr,8),
    'batch_size': batch_size,
    'max_len': max_len,
    'd_model': d_model,
    'nums_layer': nums_layer,
    'nums_head': nums_head,
    'dropout': dropout,
    'temperature': temperature,
    'top_k': top_k,
    'eval_iters': eval_iters,
    'eval_interval': eval_interval,
    'warmup_iters': warmup_iters,
}

def prepare_data():
    url_ls = ['./data/poems.ch']
    train_poems = []
    val_poems = []
    for url in url_ls:
        with open(url,'r',encoding='utf-8') as f:
            text = [line.strip() for line in f if line.strip()]

        random.shuffle(text)
        n = int(len(text) * 0.9)
        train_poems.extend(text[:n])
        val_poems.extend(text[n:])
    
    return {'train':train_poems,'val':val_poems}

text = prepare_data()

# 构建数据
data = Data(text, device=device)

# print(data.vocab_size)
# print(data.decode(data.train_data[:20].tolist()))


# xb,yb = data.get_batch('train', batch_size=1, max_len=8)
# print(data.decode(xb[0].tolist()))
# print(data.decode(yb[0].tolist()))

# 构建模型
model = GPT(data.tokenizer.vocab_size, d_model, max_len, nums_head, nums_layer, dropout)
model.to(device)

parameters = sum(p.numel() for p in model.parameters()) / 1e6
print(parameters,'M parameters')
cfg['parameters'] = parameters

# logits, loss = blm(xb,yb)
# print(logits.shape)

# idx = torch.zeros((1,1),dtype=torch.long)
# print(data.decode(blm.generate(idx,100)[0].tolist()))

# ======= 训练 =======

@torch.no_grad() # 不保存梯度，节约内存
def estimate_loss():
    model.eval() # 评估模式
    result = {}
    for split in ['train','val']:
        loss_ = []
        for k in range(eval_iters):
            X,Y = data.get_batch(split, batch_size ,max_len)
            logits,loss = model(X,Y)
            loss_.append(loss.item())
        result[split] = torch.tensor(loss_).mean().item() # 取loss的平均值
    model.train()
    return result

# 保存 loss 日志（tsv 格式，方便 excel / matplotlib 读取）和训练配置
def save_loss(loss_ls):
    with open(loss_path, 'w', encoding='utf-8') as f:
        f.write('step\ttrain_loss\tval_loss\tlr\n')
        for r in loss_ls:
            f.write(f"{r['step']}\t{r['train']:.6f}\t{r['val']:.6f}\t{r['lr']:.8f}\n")
    print(f"loss log saved to {loss_path} ({len(loss_ls)} records)")

def save_config():
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, ensure_ascii=False, indent=4)
    print(f'config saved to {config_path}')

def get_lr(step):
    if step < warmup_iters:
        # warmup
        lr = learning_rate * (step + 1) / warmup_iters
    else :
        # 余弦退火
        progress = (step - warmup_iters) / (iters - warmup_iters)
        lr = min_lr + (learning_rate - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    return lr


def train():
    save_config()
    # 优化器为 Adaw
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.1)

    loss_ls = []   # [{'step': 0, 'train': ..., 'val': ..., 'lr': ...}, ...]
    
    best_val_loss = float('inf')
    bad_count = 0

    for iter in range(iters + 1):
        lr = get_lr(iter)
        for p in optimizer.param_groups:
            p['lr'] = lr

        if iter % eval_interval == 0:
            losses = estimate_loss()
            # 保存最佳模型
            if losses['val'] < best_val_loss - early_stop_min_delta:
                best_val_loss = losses['val']
                bad_count = 0
                torch.save(model.state_dict(), ckpt_path)
            else :
                bad_count += 1
            
            loss_ls.append({
                'step': iter,
                'train': losses['train'],
                'val': losses['val'],
                'lr': lr
            })
            
            print(f"step {iter}: train loss: {losses['train']:.4f}, val loss: {losses['val']:.4f}, lr: {lr:.6f}")
            if bad_count >= early_stop_patience:
                print(f"early stopping at step {iter}")
                break
        
        if iter == iters:
            break
    
        xb, yb = data.get_batch('train', batch_size, max_len)
        logits,loss = model(xb,yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
    
    save_loss(loss_ls)

def test():
    # 加载模型权重
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.eval()

    test_poem = '忽如一夜春风来,'

    idx = torch.tensor([data.tokenizer.bos_id] + data.tokenizer.encode(test_poem)).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model.generate(idx=idx, max_new_tokens=200, 
                             temperature=0.7, 
                             top_k=40, 
                             stop_id=data.tokenizer.eos_id)
    print(data.tokenizer.decode(out[0].tolist()))

if __name__ == '__main__':
    # train()
    test()
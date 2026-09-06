import torch
import torch.nn as nn
import torch.nn.functional as F

class GPT(nn.Module):
    """
    nanoGPT模型
    """
    def __init__(self, vocab_size, d_model, max_len, nums_head=4, nums_layer=3, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_len = max_len
        # 嵌入层和位置编码层
        self.token_embedding_table = nn.Embedding(vocab_size, d_model)
        self.position_embedding_table = nn.Embedding(max_len, d_model)
        # Decoder
        self.blocks = nn.Sequential(*[Block(d_model,nums_head,dropout) for _ in range(nums_layer)])
        self.ln = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model,vocab_size)

        self.apply(self._init_weights)
        # 权重共享
        self.lm_head.weight = self.token_embedding_table.weight

    def forward(self, idx, targets = None):
        """
        token_embedding_table(idx): 将每一行中的每个词对应的词向量抽出来拼成一行，
            * 作为这一行的行向量
        """
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx) # (B,T,d_model)
        pos_emb = self.position_embedding_table(torch.arange(T, device=idx.device)) # (T,d_model)
        # 词嵌入 + 位置编码
        x = tok_emb + pos_emb # (B,T,d_model)
        # 经过Decoder
        x = self.blocks(x)
        x = self.ln(x)
        # 线性变换
        logits = self.lm_head(x) # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T, C) # cross_entropy的输入需要特定的形状
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits,targets)
        return logits,loss
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def generate(self, idx, max_new_tokens , temperature=0.8, top_k=50, stop_id=None):
        """
        idx: (B,T)
        """
        for _ in range(max_new_tokens):
            # 进行裁剪，避免将一整个句子塞入模型
            idx_cond = idx[:,-self.max_len:]

            logits,loss = self(idx_cond) # 得到预测值
            logits = logits[:,-1,:] / temperature # 取最后一个预测值，(B,C)

            if top_k is not None:
                # 找到第 K 大的 logit 值
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                # 将所有小于第 K 大值的 logits 设为负无穷，这样 softmax 后概率就是 0
                logits[logits < v[:, [-1]]] = float('-inf')

            probs = F.softmax(logits,dim = -1)
            idx_next = torch.multinomial(probs,num_samples = 1)

            if stop_id is not None and idx_next.item() == stop_id:
                break

            idx = torch.cat((idx,idx_next),dim = 1)
        return idx # (B,T + 1)
    

class Attention(nn.Module):
    """
    自注意力子层
    """
    def __init__(self, d_model, head_size, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.head_size = head_size
        self.key = nn.Linear(d_model,head_size,bias=False)
        self.query = nn.Linear(d_model,head_size,bias=False)
        self.value = nn.Linear(d_model,head_size,bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        自注意力机制
        """
        B,T,C = x.shape # (B,T,d_model)
        k = self.key(x)
        q = self.query(x)
        v = self.value(x) # (B,T,head_size)

        # out = F.scaled_dot_product_attention(
        #     q, k, v, 
        #     dropout_p=self.dropout if self.training else 0, 
        #     is_causal=True # 自动生成上三角掩码
        # )

        wei = (q @ k.transpose(-2,-1)) * (k.size(-1))**-0.5 # (B,T,T)

        #上三角掩码
        tril = torch.tril(torch.ones(T,T,device=x.device))
        wei = wei.masked_fill(tril == 0,float('-inf'))

        wei = F.softmax(wei,dim=-1) 
        wei = self.dropout(wei)
        out = wei @ v # (B,T,head_size)
        return out
    
class MultiHeadAttention(nn.Module):
    """
    多头自注意力子层:
    nums_head: 注意力头的个数
    """
    def __init__(self,d_model, nums_head, dropout=0.1):
        super().__init__()
        self.heads = nn.ModuleList([Attention(d_model,d_model // nums_head,dropout) for _ in range(nums_head)])
        self.proj = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads],dim=-1)
        out = self.dropout(self.proj(out))
        return out
    
class FeedForward(nn.Module):
    """
    前向传播: 两个线性层
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model,4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """
    Decoder块
    """
    def __init__(self, d_model, nums_head, dropout=0.1):
        super().__init__()
        # 多头自注意力
        self.sa = MultiHeadAttention(d_model,nums_head,dropout)
        # 前向传播
        self.ffn = FeedForward(d_model,dropout)
        # 层归一化
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self,x):
        """
        前置归一化, 现在的主流架构
        """
        x = x + self.sa(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x
    
    def forward_pre(self,x):
        """
        后置归一化
        """
        x = self.ln1(x + self.sa(x))
        x = self.ln2(x + self.ffn(x))
        return x

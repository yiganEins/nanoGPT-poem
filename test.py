import requests
import torch
import json
import pandas as pd
import random

# data_url = 'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt'
# data_url = 'https://github.com/sun510001/luxun_dataset/blob/master/data_dir/luxun.json'

# headers = {
#     'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36'
# }

# response = requests.get(data_url,headers=headers)
# if response.status_code == 200:
#     with open('input_b.txt','w',encoding='utf-8') as f:
#         json.dump(response.json, f, ensure_ascii=False, indent=4) 
#     print("成功")
# else:
#     print("失败")

#====== 读取json ======

# url = './data/luxun.json'
# with open(url,'r',encoding='utf-8') as f:
#     text = json.load(f)

# with open('./data/corpus.ch','w',encoding='utf-8') as f:
#     for item in text:
#         f.write(item['content'] + '\n')


# ====== 读取csv ======

# url = './datasets/poems.csv'
# df = pd.read_csv(url, encoding='utf-8')

# poems = []

# for i in df['text1'].tolist():
#     i = i.strip()
#     poems.append(i)


# ls = ['./datasets/金庸-倚天屠龙记.txt','./datasets/金庸-天龙八部.txt','./datasets/金庸-笑傲江湖.txt','./datasets/金庸-侠客行.txt']
# data = []

# for url in ls:
#     with open(url,'r',encoding='utf-8') as f:
#         ls = f.readlines()

#     text = ''
#     for i in ls:
#         i = i.strip()
#         if i:
#             data.append(i)

# with open('./data/poems_.ch','w',encoding='utf-8') as f:
#     for item in poems:
#         f.write('[诗歌]' + item + '\n')

# with open('./data/jinyong.ch','w',encoding='utf-8') as f:
#     for item in data:
#         f.write(item + '\n')


# ====== 读取txt ======

# with open('./data/poems_.txt','r',encoding='utf-8') as f:
#     text1 = f.read()

url = ['./data/luxun.ch','./data/jinyong.ch']

total_len = 0
str = ''

for u in url:
    with open(u,'r',encoding='utf-8') as f:
        text = f.read()
    str += text

print(len(str))
print(len((list(set(str)))))

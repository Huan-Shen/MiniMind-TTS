# #!/usr/bin/env python
# #-*- coding: utf-8 -*-

# import os
# import numpy as np
# import torch
# from tqdm import tqdm
# import tiktoken

# import sys
# sys.path.append("text")
# import chinese, english

# language_module_map = {"zh": chinese, "en": english}

# encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

# # onnx 模型加载完成后，开始遍历所有 wav 文件
# file_path = "H:/OpenTTSDatasets/test.txt"
# with open(file_path, "r", encoding="utf-8") as f:
#     lines = f.readlines()
#     for line in tqdm(lines):
#         line = line.strip()
#         wav_path, speaker, language, text = line.split("|")
#         language_module = language_module_map[language.lower()]
#         norm_text = language_module.text_normalize(text)

#         wav_path = wav_path.replace("E:","H:")
#         wav_path = wav_path.replace("\\", "/")
#         save_path = wav_path.replace("OpenTTSDatasets", "InputFeatures/text_token") + ".pt"
#         # if os.path.exists(save_path):
#         #     continue

#         # 将text进行tokenize
#         print(norm_text)
#         text_token = encoding.encode(norm_text)

#         # 将speech_token由list转成tensor
#         text_token = torch.tensor(text_token)     # shape: (N)
#         # 保存到指定路径
#         os.makedirs(os.path.dirname(save_path), exist_ok=True)
#         torch.save(text_token, save_path)


import os
import numpy as np
import torch
from tqdm import tqdm
import tiktoken
import sys
from multiprocessing import Pool, cpu_count

sys.path.append("./text")
import chinese, english

language_module_map = {"zh": chinese, "en": english}
encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')

file_path = "datasets/Datasets2_filelist.txt"

def process_line(line):
    line = line.strip()
    wav_path, speaker, language, text = line.split("|")
    language_module = language_module_map[language.lower()]
    norm_text = language_module.text_normalize(text)
    
    wav_path = wav_path.replace("\\", "/")
    save_path = wav_path.replace("Datasets2", "Datasets2-text_tokens") + ".pt"
    
    # 跳过已存在的文件
    if os.path.exists(save_path):
        return
    
    # 将text进行tokenize
    text_token = encoding.encode(norm_text)

    # 将speech_token由list转成tensor
    text_token = torch.tensor(text_token)  # shape: (N)
    
    # 保存到指定路径
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(text_token, save_path)
    
    return save_path

if __name__ == "__main__":
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
        total_lines = len(lines)
        
        with Pool(6) as pool:
            # 使用tqdm显示进度条
            for _ in tqdm(pool.imap_unordered(process_line, lines), total=total_lines):
                pass
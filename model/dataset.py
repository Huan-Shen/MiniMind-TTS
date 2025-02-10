import json
import random
import re

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import torch
from sklearn.model_selection import train_test_split
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

'''
制作语音合成的数据集
只需要数据集文件的路径，从文件中获取每一个输入元素所在的位置
'''


class PretrainDataset(Dataset):
    def __init__(self, file_path, max_length=1024):
        super().__init__()
        self.data = []
        self.file_path = file_path
        self.max_length = max_length
        self.padding = 0

        # 加载文本
        with open(file_path, "r", encoding="utf-8") as f:
            filelist = f.readlines()
            for line in filelist:
                line = line.strip()
                audio_path, speaker, lung, text = line.split("|")
                self.data.append((audio_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index: int):
        # 1. 按照索引读取 self.data
        sample = self.data[index]

        # 2. 将其转换成 text_token_path 和 speech_token_path
        text_token_path = sample.replace("Datasets2", "Datasets2-text_tokens") + ".pt"
        speech_token_path = sample.replace("Datasets2", "Datasets2-speech_tokens") + ".pt"

        # 3. 读取 text_token_path 和 speech_token_path 获取对应的token
        text_token = torch.load(text_token_path).cpu()
        speech_token = torch.load(speech_token_path)  # 这里读出来的维度是 【1，1，T】，我想要的是【T】
        speech_token = torch.squeeze(torch.squeeze(speech_token, dim = 0 ), dim=0).cpu()

        # 4. 获取 text_token 和 speech_token 的长度
        text_len = len(text_token)
        speech_len = len(speech_token)

        # 没满最大长度的text_token剩余部分
        padding_text_token_len = self.max_length - text_len
        text_token = F.pad(text_token, (0, padding_text_token_len), value=0)

        # 没满最大长度的speech_token剩余部分
        padding_speech_token_len = self.max_length - speech_len
        speech_token = F.pad(speech_token, (0, padding_speech_token_len), value=0)

        return text_token, text_len, speech_token, speech_len



if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader
    import os

    # 设置测试文件路径
    test_file_path = "datasets/Datasets2_filelist.txt"

    # 创建数据集实例
    dataset = PretrainDataset(test_file_path, max_length=64)  # 设置 max_length 以测试 padding

    # 使用 DataLoader 测试批量加载
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)

    # 遍历 DataLoader 检查结果
    for text_token, text_len, speech_token, speech_len in dataloader:
        print("Input (X):", text_token)
        print("Output (Y):", speech_token)
        print("X shape:", text_token.shape)
        print("Y shape:", speech_token.shape)

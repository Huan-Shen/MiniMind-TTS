import os
import sys
import torchaudio
import random
import time

import numpy as np
import torch
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import Transformer
from model.LMConfig import LMConfig
import tiktoken

sys.path.append(r"WavTokenizer")
from decoder.pretrained import WavTokenizer

warnings.filterwarnings('ignore')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def init_model(lm_config):
    moe_path = '_moe' if lm_config.use_moe else ''
    ckp = f'./out/pretrain_{lm_config.dim}{moe_path}.pth'

    model = Transformer(lm_config)
    state_dict = torch.load(ckp, map_location=device)

    # 处理不需要的前缀
    unwanted_prefix = '_orig_mod.'
    for k, v in list(state_dict.items()):
        if k.startswith(unwanted_prefix):
            state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

    for k, v in list(state_dict.items()):
        if 'mask' in k:
            del state_dict[k]

    # 加载到模型中
    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)

    print(f'模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')

    wavtokenizer = WavTokenizer.from_pretrained0802("./WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
                                                    "./WavTokenizer/WavTokenizer_small_320_24k_4096.ckpt")
    wavtokenizer = wavtokenizer.to(device)
    return model, wavtokenizer


def setup_seed(seed):
    random.seed(seed)  # 设置 Python 的随机种子
    np.random.seed(seed)  # 设置 NumPy 的随机种子
    torch.manual_seed(seed)  # 设置 PyTorch 的随机种子
    torch.cuda.manual_seed(seed)  # 为当前 GPU 设置随机种子（如果有）
    torch.cuda.manual_seed_all(seed)  # 为所有 GPU 设置随机种子（如果有）
    torch.backends.cudnn.deterministic = True  # 确保每次返回的卷积算法是确定的
    torch.backends.cudnn.benchmark = False  # 关闭 cuDNN 的自动调优，避免不确定性


if __name__ == "__main__":
    # -----------------------------------------------------------------------------
    out_dir = 'out'
    start = ""
    temperature = 0.7
    top_k = 8
    setup_seed(1337)
    # device = 'cpu'
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    dtype = 'bfloat16'
    max_seq_len = 2048
    lm_config = LMConfig()
    lm_config.max_seq_len = max_seq_len
    # -----------------------------------------------------------------------------
    encoding = tiktoken.encoding_for_model('gpt-3.5-turbo')
    model, wavtokenizer = init_model(lm_config)
    model = model.eval()

    text = "hello, world!"
    text_token = encoding.encode(text)
    text_token = torch.tensor(text_token).to(device)  # shape: (N)

    output = model.generate(text_token, max_new_tokens=50, temperature=0.8, top_k=5)  # [1, S]

    output = output.to(device)
    features2 = wavtokenizer.codes_to_features(output)
    bandwidth_id = torch.tensor([0]).to(device) 
    audio_out = wavtokenizer.decode(features2, bandwidth_id=bandwidth_id)

    # 将tensor转成numpy
    audio_out = audio_out.cpu()
    torchaudio.save("audio_out.wav", audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)
from model.model import Transformer
from model.LMConfig import LMConfig
import torch

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    config = LMConfig()
    model = Transformer(config)
    print(model)
    print(f'\n模型参数: {count_parameters(model) / 1e6} 百万 = {count_parameters(model) / 1e9} B (Billion)')

    text_tokens = torch.randint(0, config.vocab_size, (4, 1024))
    text_token_len = torch.tensor([15, 5, 8, 21])
    print(f'\n输入: {text_tokens.shape}')

    speech_tokens = torch.randint(0, config.speechcab_size, (4, 1024))
    speech_token_len = torch.tensor([24, 34, 27, 26])
    print(f'\n输入: {speech_tokens.shape}')

    outputs = model(text_tokens, text_token_len, speech_tokens, speech_token_len)
    print(f'\n输出: {outputs[0].shape}')
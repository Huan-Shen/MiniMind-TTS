# --coding:utf-8--
import os

from encoder.utils import convert_audio
import torchaudio
import torch
from decoder.pretrained import WavTokenizer

wavtokenizer = WavTokenizer.from_pretrained0802("configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml", "WavTokenizer_small_320_24k_4096.ckpt")
wavtokenizer = wavtokenizer.cuda()

wav, sr = torchaudio.load("E:/Datasets/Datasets2/kefu001/000012.wav")
bandwidth_id = torch.tensor([0])
wav=wav.cuda()
print(wav)

features,discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id.cuda())
print(discrete_code)


audio_out = wavtokenizer.decode(features.cuda(), bandwidth_id=bandwidth_id.cuda())
# 将tensor转成numpy
audio_out = audio_out.cpu()
torchaudio.save("audio_out.wav", audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)


features2 = wavtokenizer.codes_to_features(discrete_code)
bandwidth_id = torch.tensor([0])  
audio_out = wavtokenizer.decode(features2, bandwidth_id=bandwidth_id)

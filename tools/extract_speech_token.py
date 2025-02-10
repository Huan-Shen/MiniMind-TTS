# # --coding:utf-8--
# import os
# import sys
# import torchaudio
# import torch
# from tqdm import tqdm 

# sys.path.append(r"WavTokenizer")
# from encoder.utils import convert_audio
# from decoder.pretrained import WavTokenizer


# # 1. 加载音频量化模型
# wavtokenizer = WavTokenizer.from_pretrained0802("./WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml", 
#                                                 "./WavTokenizer/WavTokenizer_small_320_24k_4096.ckpt")
# wavtokenizer = wavtokenizer.cuda()

# # 2. 读取音频数据集文件
# audio_paths = []
# with open("./datasets/Datasets2_filelist.txt", "r", encoding="utf-8") as f:
#     filelist = f.readlines()
#     for line in filelist:
#         line = line.strip()
        
#         audio_path, speaker, lung, text = line.split("|")
#         audio_paths.append(audio_path)

# # 3. 逐个音频文件进行量化
# for audio_path in tqdm(audio_paths):
#     wav, sr = torchaudio.load(audio_path)
#     bandwidth_id = torch.tensor([0])
#     wav=wav.cuda()
#     print(wav)

#     save_path = audio_path.replace("Datasets2", "Datasets2-speech_tokens") + ".pt"
#     features, discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id.cuda())
    
#     # 4. 保存量化结果
#     torch.save(discrete_code, save_path)
    

#     # 解码
#     # audio_out = wavtokenizer.decode(features.cuda(), bandwidth_id=bandwidth_id.cuda())
#     # # 将tensor转成numpy
#     # audio_out = audio_out.cpu()
#     # torchaudio.save("audio_out.wav", audio_out, sample_rate=24000, encoding='PCM_S', bits_per_sample=16)


#     # features2 = wavtokenizer.codes_to_features(discrete_code)
#     # bandwidth_id = torch.tensor([0])  
#     # audio_out = wavtokenizer.decode(features2, bandwidth_id=bandwidth_id)


# 多线程处理
# --coding:utf-8--
import os
import sys
import torchaudio
import torch
from tqdm import tqdm
import threading
import concurrent.futures  # 导入线程池模块

sys.path.append(r"WavTokenizer")
from encoder.utils import convert_audio
from decoder.pretrained import WavTokenizer


# 1. 加载音频量化模型 (只需加载一次)
wavtokenizer = WavTokenizer.from_pretrained0802("./WavTokenizer/configs/wavtokenizer_smalldata_frame75_3s_nq1_code4096_dim512_kmeans200_attn.yaml",
                                                "./WavTokenizer/WavTokenizer_small_320_24k_4096.ckpt")
wavtokenizer = wavtokenizer.cuda()
wavtokenizer.share_memory() # 重要: 为了在多线程中共享模型，需要调用 share_memory()


# 2. 读取音频数据集文件
audio_paths = []
with open("./datasets/Datasets2_filelist.txt", "r", encoding="utf-8") as f:
    filelist = f.readlines()
    for line in filelist:
        line = line.strip()
        audio_path, speaker, lung, text = line.split("|")
        if speaker != "BZNSYP":
            continue
        # 如果文件存在，则添加到列表中
        if os.path.exists(audio_path):
            audio_paths.append(audio_path)

        save_path = audio_path.replace("Datasets2", "Datasets2-speech_tokens") + ".pt"
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

# 定义一个函数来处理单个音频文件
def process_audio(audio_path):
    try: # 增加 try-except 块来捕获单个线程中的错误，避免影响其他线程
        wav, sr = torchaudio.load(audio_path)
        bandwidth_id = torch.tensor([0])
        wav=wav.cuda() # 将wav数据移动到GPU上, 注意每个线程都需要移动数据到自己的GPU context中
        features, discrete_code= wavtokenizer.encode_infer(wav, bandwidth_id=bandwidth_id.cuda())

        # 4. 保存量化结果
        save_path = audio_path.replace("Datasets2", "Datasets2-speech_tokens") + ".pt"
        # 如果文件夹不存在，创建文件夹
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(discrete_code, save_path)

        return True # 返回 True 表示处理成功
    except Exception as e:
        print(f"Error processing {audio_path}: {e}") # 打印错误信息
        return False # 返回 False 表示处理失败


# 3. 使用线程池进行多线程处理
num_threads = 6  # 可以根据CPU核心数和GPU负载调整线程数，例如设置为 8 或者 16
success_count = 0
failed_count = 0

with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
    futures = [executor.submit(process_audio, path) for path in audio_paths] # 提交任务到线程池
    for future in tqdm(concurrent.futures.as_completed(futures), total=len(audio_paths)): # 使用as_completed来跟踪进度
        if future.result(): # 获取process_audio函数的返回值
            success_count += 1
        else:
            failed_count += 1

print(f"Total processed: {len(audio_paths)}, Success: {success_count}, Failed: {failed_count}")
print("All audio files processed in multi-threading.")
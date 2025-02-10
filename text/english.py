import pickle
import os
import re
from g2p_en import G2p
import LangSegment
import string
punc = string.punctuation

import symbols
from en_norm import cleaners

current_file_path = os.path.dirname(__file__)
CMU_DICT_PATH = os.path.join(current_file_path, "cmudict.rep")
CMU_DICT_FAST_PATH = os.path.join(current_file_path, "cmudict-fast.rep")
CMU_DICT_HOT_PATH = os.path.join(current_file_path, "engdict-hot.rep")
CACHE_PATH = os.path.join(current_file_path, "engdict_cache.pickle")
CHINESE_DICT_PATH = os.path.join(current_file_path, "Chinese_dict.txt")
_g2p = G2p()

arpa = {
    "AH0",
    "S",
    "AH1",
    "EY2",
    "AE2",
    "EH0",
    "OW2",
    "UH0",
    "NG",
    "B",
    "G",
    "AY0",
    "M",
    "AA0",
    "F",
    "AO0",
    "ER2",
    "UH1",
    "IY1",
    "AH2",
    "DH",
    "IY0",
    "EY1",
    "IH0",
    "K",
    "N",
    "W",
    "IY2",
    "T",
    "AA1",
    "ER1",
    "EH2",
    "OY0",
    "UH2",
    "UW1",
    "Z",
    "AW2",
    "AW1",
    "V",
    "UW2",
    "AA2",
    "ER",
    "AW0",
    "UW0",
    "R",
    "OW1",
    "EH1",
    "ZH",
    "AE0",
    "IH2",
    "IH",
    "Y",
    "JH",
    "P",
    "AY1",
    "EY0",
    "OY2",
    "TH",
    "HH",
    "D",
    "ER0",
    "CH",
    "AO1",
    "AE1",
    "AO2",
    "OY1",
    "AY2",
    "IH1",
    "OW0",
    "L",
    "SH",
}


def replace_phs(phs):
    rep_map = {
        ";": ",", 
        ":": ",", 
        "'": "-", 
        '"': "-",
        "：": ",",
        "；": ",",
        "，": ",",
        "。": ".",
        "！": "!",
        "？": "?",
        "\n": ".",
        "·": ",",
        "、": ",",
        "...": "…",
        }
    phs_new = []
    for ph in phs:
        if ph in symbols:
            phs_new.append(ph)
        elif ph in rep_map.keys():
            phs_new.append(rep_map[ph])
        else:
            print("ph not in symbols: ", ph)
    return phs_new


def read_dict():
    g2p_dict = {}
    start_line = 49
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    return g2p_dict


def read_dict_new():
    g2p_dict = {}
    g2p_zh_dict = {}
    with open(CMU_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 49:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                syllable_split = word_split[1].split(" - ")
                g2p_dict[word] = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    g2p_dict[word].append(phone_split)

            line_index = line_index + 1
            line = f.readline()

    with open(CMU_DICT_FAST_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0]
                if word not in g2p_dict:
                    g2p_dict[word] = []
                    g2p_dict[word].append(word_split[1:])

            line_index = line_index + 1
            line = f.readline()

    with open(CMU_DICT_HOT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= 0:
                line = line.strip()
                word_split = line.split(" ")
                word = word_split[0]
                #if word not in g2p_dict:
                g2p_dict[word] = []
                g2p_dict[word].append(word_split[1:])

            line_index = line_index + 1
            line = f.readline()
    
    # 添加中文地名
    start_line = 1
    with open(CHINESE_DICT_PATH) as f:
        line = f.readline()
        line_index = 1
        while line:
            if line_index >= start_line:
                line = line.strip()
                word_split = line.split("  ")
                word = word_split[0]

                # print(word)
                syllable_split = word_split[1].split(" - ")
                g2p_zh_dict[word] = []
                for syllable in syllable_split:
                    if " " in syllable:
                        phone_split = syllable.split(" ")
                        g2p_zh_dict[word].append(phone_split)
                    else:
                        g2p_zh_dict[word].append(phone_split)   

            line_index = line_index + 1
            line = f.readline()


    return g2p_dict, g2p_zh_dict


def cache_dict(g2p_dict, file_path):
    with open(file_path, "wb") as pickle_file:
        pickle.dump(g2p_dict, pickle_file)


def get_dict():
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH, "rb") as pickle_file:
            g2p_dict = pickle.load(pickle_file)
    else:
        g2p_dict = read_dict_new()
        cache_dict(g2p_dict, CACHE_PATH)

    return g2p_dict


def text_normalize(text):
    # todo: eng text normalize
    text = cleaners.english_cleaners(text)
    return text

letters = [chr(ord('A') + i) for i in range(26)]
special_words = ["PCIe","GTX","FHD","HDD","SSD", "DDG","XDR", "ID", "LG","RPG", "ROG", "OLED", "LED","PS","PR","HDR","ARGB", "RTX", "CPU", "GPU", "CUDA", "DPI", "USB", "HDMI", "PD", "IPS","AMD","AI","ATX","LCD","DVD","EVD","PCIE","AIGC","NLP","CNN","SVM","RNN","LSTM","NPU","KF","HK"]
def is_alphabet( char):
    """判断一个unicode是否是英文"""
    if (char >= '\u0041' and char <= '\u005a') or (char >= '\u0061' and
                                                    char <= '\u007a'):
        return True
    else:
        return False

def g2p(text, new_word_dict):
    eng_dict, zh_dict = read_dict_new()
    # 把英文单词拆成单个字符
    norm_text = ""
    types = []
    for ch in text:
        if is_alphabet(ch):
            types.append((ch, "en"))
        elif ch == "'":
            types.append((ch, "en"))
        else:
            types.append((ch, "other"))
    # 2. 先按照顺序，把同种类的合并。
    segments = []
    if types:
        current_type = types[0][1]
        current_segment = types[0][0]
        
        for i in range(1, len(types)):
            char, typ = types[i]
            if typ == current_type:
                current_segment += char
            else:
                segments.append((current_segment, current_type))
                current_segment = char
                current_type = typ
                
        segments.append((current_segment, current_type))
    # print(segments)

    # 遍历 segments， 转换成 phones
    phones = []
    for i in range(len(segments)):
        txt = segments[i][0].replace(" ", "")
        if(len(txt) == 0):
            txt = " "
        typ = segments[i][1]
        
        if typ == "en":
            # 首先判断它是否是单个字母
            if txt in letters:
                txt = txt + '.'   # 单个单词，都需要加上.才能念对

            if txt in special_words:
                for t in txt:
                    norm_text = norm_text + t.replace(".","") + " "

                    if t.upper() in letters:
                        t = t.upper() + '.'   # 单个单词，都需要加上.才能念对
                        phns = eng_dict[t.upper()]
                        for ph in phns:
                            phones += ph

            elif txt.upper() in new_word_dict:   # 如果新添加的字典里存在此单词
                p = new_word_dict[txt.upper()]
                if "-" in p:
                    syllable_split = p.split(" - ")
                else:
                    syllable_split = [p]

                phns = []
                for syllable in syllable_split:
                    phone_split = syllable.split(" ")
                    phns.append(phone_split)

                for ph in phns:
                    phones += ph
                norm_text = norm_text + txt.replace(".","")

            elif txt.upper() in eng_dict:   # 如果原始大字典里存在此单词
                phns = eng_dict[txt.upper()]
                for ph in phns:
                    phones += ph

                norm_text = norm_text + txt.replace(".","")

            elif txt.upper() in zh_dict:   # 如果中文地名的字典里存在此单词
                phns = zh_dict[txt.upper()]
                for ph in phns:
                    phones += ph

                norm_text = norm_text + txt.replace(".","")
            else:
                # 通过LangSegment筛选符合条件的
                formattext = ""
                for tmp in LangSegment.getTexts(txt):
                    formattext += tmp["text"] + " "
                    while "  " in formattext:
                        formattext = formattext.replace("  ", " ")

                text_seg = re.split(" ", formattext)
                for tx in text_seg:
                    if len(tx)==0 or (len(formattext.strip()) == 0):
                        continue
                    if tx in letters:
                        tx = tx + '.'   # 单个单词，都需要加上.才能念对
                        
                    if tx.upper() in eng_dict:
                        phns = eng_dict[tx.upper()]
                        for ph in phns:
                            phones += ph
                        norm_text = norm_text + tx.replace(".","") + " "

                    else:  # 如果还不在字典里，那就拆成单个单词
                        if len(tx)>0:
                            for t in tx:
                                norm_text = norm_text + t.replace(".","") + " "

                                if t.upper() in letters:
                                    t = t.upper() + '.'   # 单个单词，都需要加上.才能念对
                                    phns = eng_dict[t.upper()]
                                    for ph in phns:
                                        phones += ph 

                        else:
                            continue

        elif typ != "en" and txt==" ":
            norm_text = norm_text + txt
            continue

        else:
            txt = txt.replace(" ", "")
            phones += txt   
            norm_text = norm_text + txt

    return replace_phs(phones), norm_text


if __name__ == "__main__":
    from symbols import symbols
    from en_norm import cleaners

    print(g2p(text_normalize("In this; paper, we propose 1 DSPGAN, a GAN-based universal vocoder. ChatTTS, is .ganbase.")))


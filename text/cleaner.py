import chinese, cleaned_text_to_sequence, symbols, english
import string
import re

language_module_map = {"zh": chinese, "en": english}


def clean_text(text, language, new_word_dict):
    if(language not in language_module_map):
        language="en"
        text=" "

    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    # print("norm_text: ", norm_text)

    ## 检测连续的标点符号
    # 获取所有标点符号
    punctuation = r"""!"！#&'‘’“”()（）*+,，-·——.。/、:：;；<《》=>?？@……【】[\]^_…`{|}~"""
    # 处理后的文本
    new_text = ""
    # 上一个字符是否为标点符号
    prev_is_punctuation = False
    for char in norm_text:
        if char in punctuation:
            if not prev_is_punctuation:
                new_text += char
                prev_is_punctuation = True
        else:
            new_text += char
            prev_is_punctuation = False

    norm_text = new_text
    # print("norm_text: ", norm_text)

    if language == "zh":
        phones, word2ph = language_module.g2p(norm_text)
        assert len(phones) == sum(word2ph)
        assert len(norm_text) == len(word2ph)
    else:
        phones, norm_text = language_module.g2p(norm_text, new_word_dict)
        word2ph = None

    for ph in phones:
        assert ph in symbols
    return phones, word2ph, norm_text


def clean_special(text, language, special_s, target_symbol):
    """
    特殊静音段sp符号处理
    """
    text = text.replace(special_s, ",")
    language_module = language_module_map[language]
    norm_text = language_module.text_normalize(text)
    phones = language_module.g2p(norm_text)
    new_ph = []
    for ph in phones[0]:
        assert ph in symbols
        if ph == ",":
            new_ph.append(target_symbol)
        else:
            new_ph.append(ph)
    return new_ph, phones[1], norm_text


def text_to_sequence(text, language):
    phones = clean_text(text)
    return cleaned_text_to_sequence(phones)


if __name__ == "__main__":
    print(clean_text("你好%啊啊啊额、还是到付红四方。", "zh"))

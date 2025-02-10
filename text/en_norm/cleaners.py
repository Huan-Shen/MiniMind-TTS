'''
Cleaners are transformations that run over the input eng_text_norm at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English eng_text_norm
  2. "transliteration_cleaners" for non-English eng_text_norm that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
'''

import re
# from unidecode import unidecode

from .numbers import normalize_numbers
from .symbols import _characters as vocab


# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\s+{0}\s+'.format(x[0]), re.IGNORECASE), x[1]) for x in [
    ('mrs.', 'misess '),
    ('mr.', 'mister '),
    ('dr.', 'doctor'),
    # ('st.', 'saint'),
    # ('co.', 'company'),
    # ('jr.', 'junior'),
    # ('maj.', 'major'),
    # ('gen.', 'general'),
    ('drs.', 'doctors'),
    # ('rev.', 'reverend'),
    # ('lt.', 'lieutenant'),
    # ('hon.', 'honorable'),
    # ('sgt.', 'sergeant'),
    # ('capt.', 'captain'),
    # ('esq.', 'esquire'),
    # ('ltd.', 'limited'),
    # ('col.', 'colonel'),
    # ('ft.', 'fort'),
    ('e.g.', 'for example, '),
    ('i.e.', 'in essense, '),
    ('etc.', 'and so on. '),
    # ('p.s.', 'post scripts, '),
]]


def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


# List of english symbolic expressions:
_symbolic_expressions = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('\\\\~/', ''),
    ('\^5', ''),
    ('\?\^', ''),
    ('\\\\_/\?', ''),
    ('\[_\]>', ''),
    ('@@@', ''),
    ('@--/--', ''),
    (':\)', ''),
    ('0:-\)', ''),
    ('!:-\)', ''),
    (':-\)', ''),
    (';-\)~', ''),
    (';-\)', ''),
    ('<:-\|', ''),
    (':~\)', ''),
    ('>:-\(', ''),
    (':-\(', ''),
    ('#8-\)', ''),
    ('8-\)', ''),
    (':-}', ''),
    (':-/', ''),
    (':,-\(', ''),
    (':-<', ''),
    ('>:-\|\|', ''),
    (':-\|', ''),
    (':->', ''),
    ('\|-\)', ''),
    ('=:-0', ''),
    (':-0', ''),
    (':-o', ''),
    (':-D', ''),
    (':-x', ''),
    (':o\)', ''),
    (';-P', ''),
    (':P', ''),
    ('&-\(', ''),
    (';-{\)', ''),
    (';~\)', ''),
    ('c\["\]', ''),
    ('<:-|', ''),
    ('\|-\|', ''),
]]


def replace_symbolic_expressions(text):
    for regex, replacement in _symbolic_expressions:
        text = re.sub(regex, replacement, text)
    return text


# # List of english symbolic acronym:
# _symbolic_acronyms = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
#     ('can\'t', 'can not'),
#     ('cannot', 'can not'),
#     ('what\'s', 'what is'),
#     ('\'ve ', ' have '),
#     ('n\'t ', ' not '),
#     ('i\'m', 'i am'),
#     ('\'re ', ' are '),
#     ('he\'s ', ' he is '),
#     ('she\'s ', ' she is '),
#     ('it\'s ', ' it is '),
#     ('that\'s ', ' that is '),
#     ('there\'s ', ' there is '),
#     ('\'d ', ' would '),
#     ('\'ll ', ' will '),
#     ('(\d+\.?\d+)l', 'length \\1'),
#     ('(\d+\.?\d+)w', 'width \\1'),
#     ('(\d+\.?\d+)h', 'height \\1')
# ]]


# def expand_acronyms(text):
#     for regex, replacement in _symbolic_acronyms:
#         text = re.sub(regex, replacement, text)
#     return text


# Remove ordinal:
_ordinal_numbers = [(re.compile('\s*%s\s*' % x[0], re.IGNORECASE), x[1]) for x in [
    ('\(\d\)', ' '),
    ('\[\d\]', ' '),
    ('\<\d\>', ' '),
    ('\{\d\}', ' '),
    ('\([a-f]\)', ' '),
    ('\s[a-f]\.', ' '),
    ('\([iv]+\)', ' '),
]]


def remove_ordinal(text):
    for regex, replacement in _ordinal_numbers:
        text = re.sub(regex, replacement, text)
    return text


# Process english punctuations:
_punctuations = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('，', ','),
    ('；', ';'),
    ('。', '.'),
    ('！', '!'),
    ('？', '?'),
    ('[‘’]', '\''),
    ('[“”]', '\"'),
    ('（', '('),
    ('）', ')'),
    ('\"', ''),
    ('\' ', ' '),
    (' \'', ' '),
    ('\s*[.]+\s*', ' . '),
    ('\s*[!]+\s*', ' ! '),
    ('\s*[?]+\s*', ' ? '),
    ('\s*[-]{2,}\s*', ' , '),
    ('\s+-', ' , '),
    ('-\s+', ' , '),
    ('\s*,\s*', ' , '),
    ('\s*;\s*', ' ; '),
    ('[()]', ' '),
    ('\s*/\s*', ' '),
]]


def normalize_punctuations(text):
    for regex, replacement in _punctuations:
        text = re.sub(regex, replacement, text)
    return text


_operators = [(re.compile('%s' % x[0], re.IGNORECASE), x[1]) for x in [
    ('\s+[x*]\s+', ' times '),
    ('\s*\+\s*', ' plus '),
    ('\s*=\s*', ' equal '),
    ('\s*&\s*', ' and '),
    ('\s*\|\s*', ' or '),
    ('\s*/\s*', ' or '),
]]


def normalize_operators(text):
    for regex, replacement in _operators:
        text = re.sub(regex, replacement, text)
    return text


def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def remove_out_vocab(text):
    return re.sub('[^{}]'.format(vocab), ' ', text)


# def convert_to_ascii(text):
#     return unidecode(text)


def basic_cleaners(text):
    '''Basic pipeline that lowercases and collapses whitespace without transliteration.'''
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    '''Pipeline for non-English eng_text_norm that transliterates to ASCII.'''
    # text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    '''Pipeline for English eng_text_norm, including number and abbreviation expansion.'''
    # # text = convert_to_ascii(text)
    # text = replace_symbolic_expressions(text)
    # # text = lowercase(text)
    # text = remove_ordinal(text)
    # # text = expand_acronyms(text)
    # text = expand_abbreviations(text)
    # text = expand_numbers(text)
    # text = normalize_operators(text)
    text = normalize_punctuations(text)
    text = collapse_whitespace(text)
    text = remove_out_vocab(text)
    text = text.strip()
    # if text[0] == "'":
    #     text = text.replace("'", "", 1)
    # text = text.replace("-", " ")
    # text = text.replace("  ", " ")
    # text = text.replace(';', ',')
    # text = text.replace(':', ',')
    # text = text.replace('?', ',')
    # text = text.replace('!', ',')

    return text

if __name__ == "__main__":
    from numbers import normalize_numbers
    from symbols import _characters as vocab

    text0 =' mister wang said "why, it\'s impossible to carry that to camden town," said scrooge.\" mister han'
    text = english_cleaners(text0)
    print(text)



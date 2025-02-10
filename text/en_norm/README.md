# Eng Text Norm
this is a repository for english text normalization (no longer maintained).

> Log: use [Joee1995/fraction_to_text](https://github.com/Joee1995/fraction_to_text.git) resolve the bug of coverting fraction to english text.

## Quick Start ##

### Git Clone Repo ###
git clone this repo to the root directory of your project which need to use it.

    cd /path/to/proj
    git clone https://github.com/Joee1995/eng-text-norm.git

after that, your doc tree should be:
```
proj                     # root of your project
|--- eng_text_norm       # this chn-text-norm tool
     |--- cleaners.py
     |--- ...
|--- text_normalize.py   # your text normalization code
|--- ...
```

### How to Use ? ###

    # text_normalize.py
    from eng_text_norm import cleaners
    
    raw_text = 'your raw text'
    text = clearners.english_cleaners(raw_text)


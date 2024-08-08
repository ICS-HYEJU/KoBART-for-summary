import jsonlines
import json
import time
import re
import pandas as pd
import itertools


from tqdm import tqdm
from ast import literal_eval
from transformers import PreTrainedTokenizerFast
import sentencepiece as spm

def split_file():
    path = '/storage/hjchoi/Document_Summary_text/Training/news_train_original/train_original.json'
    make_path = '/storage/hjchoi/Document_Summary_text/Training/news_train_split/'
    with open(path, 'r') as f:
        json_file = json.load(f)
        doc = json_file['documents']
        for i in range(len(doc)):
            with open(make_path + str(i) + '.json', 'w', encoding='UTF-8') as f:
                f.write(json.dumps(doc[i], ensure_ascii=False))

# We can use this function as a getitem function.
# Remove Stopwords
def read_file():
    from config.config import get_config_dict
    cfg = get_config_dict()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.dataset_info['pretrained_name'])
    path = '/storage/hjchoi/Document_Summary_text/Training/news_train_split/0.json'
    with open(path, 'r') as f:
        data = json.load(f)
        print(data)
        text = list(itertools.chain(*data['text']))
        # remove stopwords
        sentence = []
        line = ''
        for id in tqdm(range(len(text)), desc="remove stopwords from origin", mininterval=0.01):
            stop_idx = re.split(r'[,;]', text[id]['highlight_indices'])
            for i, v in enumerate(text[id]['sentence']):
                if str(i) in stop_idx:
                    continue
                line += v
            sentence.append(line)
            line = ''
            time.sleep(0.1)
        print(sentence)


if __name__ == '__main__':
    # split_file()
    read_file()


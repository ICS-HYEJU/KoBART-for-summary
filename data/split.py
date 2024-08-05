import jsonlines
import json
import pandas as pd
from tqdm import tqdm
from ast import literal_eval

def split_file():
    path = '/storage/hjchoi/Document_Summary_text/Training/news_train_original/train_original.json'
    line_cnt = 0
    with open(path, 'r') as f:
        json_file = json.load(f)
        doc = json_file['documents']
        for i in range(len(doc)):
            with open(path + '_' + str(i) + '.json', 'w', encoding='UTF-8-sig') as f:
                f.write(json.dumps(doc[i], ensure_ascii=False))



if __name__ == '__main__':
    split_file()

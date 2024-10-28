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
    path = '/storage/hjchoi/Document_Summary_text/Validation/magazine_valid_original/valid_original.json'
    make_path = '/storage/hjchoi/Document_Summary_text/Validation/magazine_valid_split/'
    with open(path, 'r') as f:
        json_file = json.load(f)
        doc = json_file['documents']
        for i in tqdm(range(len(doc)),desc='splitting and delete None abstractive data'):
            if doc[i]['abstractive'][0] == "":
                print(f"i:{i} , abstractive : {doc[i]['abstractive']}")
                continue
            else:
                with open(make_path + str(i) + '.json', 'w', encoding='UTF-8') as f:
                    f.write(json.dumps(doc[i], ensure_ascii=False))

# We can use this function as a getitem function.
# Remove Stopwords
# def read_file():
#     from config.config import get_config_dict
#     cfg = get_config_dict()
#     tokenizer = PreTrainedTokenizerFast.from_pretrained(cfg.dataset_info['pretrained_name'])
#     #
#     for jf in tqdm(range(243983), desc='Remake data[text]', mininterval=0.01):
#         path = '/storage/hjchoi/Document_Summary_text/Training/news_train_split/'
#         with open(path+str(jf)+'.json', 'r') as f:
#             data = json.load(f)
#             text = list(itertools.chain(*data['text']))
#             text_line = []
#             make_data = []
#             # remove stopwords
#             line = ''
#             for id in tqdm(range(len(text)), desc="remove stopwords from origin", mininterval=0.01):
#                 stop_idx = re.split(r'[,;]', text[id]['highlight_indices'])
#                 for i, v in enumerate(text[id]['sentence']):
#                     if str(i) in stop_idx:
#                         continue
#                     line += v
#                 text[id]['sentence'] = [tokenizer.bos_token_id] + tokenizer.encode(line) + [tokenizer.eos_token_id]
#                 del text[id]['highlight_indices']
#                 text[id]['summary'] = tokenizer.encode(data['abstractive'][0])
#                 make_data.append(text[id])
#                 line = ''
#             with open('/storage/hjchoi/Document_Summary_text/Training/news/' + str(jf)+'.json', 'w', encoding='UTF-8') as f:
#                 f.write(json.dumps(make_data, ensure_ascii=False))
#             make_data = []
#             time.sleep(0.1)
#         time.sleep(0.1)

def json_to_pandas():
    from config.config import get_config_dict
    cfg = get_config_dict()
    origin = []
    summary = []
    index=[]
    #
    for jf in tqdm(range(7008), desc='Remake data[text]', mininterval=0.01):
        path = '/storage/hjchoi/Document_Summary_text/Validation/magazine_valid_split/'
        if jf == 1187:
            jf += 1
        with open(path+str(jf)+'.json', 'r') as f:
            data = json.load(f)
            if data['abstractive'] is None:
                print(f'{jf}.json file does not contain abstractive')
                print(data['abstractive'])
            text = list(itertools.chain(*data['text']))

            # remove stopwords
            line = ''
            for id in tqdm(range(len(text)), desc="remove stopwords from origin", mininterval=0.01):
                stop_idx = re.split(r'[,;]', text[id]['highlight_indices'])
                for i, v in enumerate(text[id]['sentence']):
                    if str(i) in stop_idx:
                        continue

                    line += v
            origin.append(line)
            summary.append(data['abstractive'][0])
            index.append(data['id'])

    df = pd.DataFrame({'id':index, 'text':origin,'summary':summary})
    # Remove None value
    df.dropna(how='any', inplace=True)
    df.to_csv('/storage/hjchoi/Document_Summary_text/Validation/magazine.tsv', sep='\t', index=False)
    time.sleep(0.1)

if __name__ == '__main__':
    # split_file()
    # read_file()
    json_to_pandas()

import numpy as np
from bs4 import BeautifulSoup
from datetime import datetime
import requests
import pandas as pd
import re
import time
import sys, os
from tqdm import tqdm
import os
import io
import streamlit as st
import torch

from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
import lightning as L
#
from model.bart import KoBARTGeneration
from config.config import get_config_dict
from model.bart import KoBARTGeneration
from config.config import get_config_dict


cfg = get_config_dict()
device = torch.device('cuda:{}'.format(1))
torch.cuda.set_device(1)

def crawling_main_text(url:str):
    response = requests.get(url)
    response.encoding = None
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    # YONHAP_NEWS

    if ('yna' in url) | ('app.yonhapnews' in url):
        try:
            main_article = soup.find('article', {'class': 'story-news article'})
            text = []
            for s in main_article.find_all('p'):
                text.append(s.get_text())

            if main_article == None:
                main_article = soup.find('div', {'class': 'article-txt'})

            text = main_article.text
        except:
            return

    # MBC_NEWS
    elif ('imnews.imbc' in url) or ('mnews' in url):
        try:
            text = soup.find('div', {'itemprop': 'articleBody'})
            if text is None:
                text = soup.find('div', {'id':'newsct_article'})
            text = text.text
        except:
            return

    # MAEIL_NEWS(Miracle)
    elif 'mirakle.mk' in url:
        try:
            text = soup.find('div', {'class': 'view_txt'}).text
        except:
            return

    # MAEIL_NEWS
    elif ('mk.co' in url) or ('imaeil' in url):
        try:
            text = soup.find('div', {'itemprop': 'articleBody'}).text
        except:
            return

    # SBS_NEWS
    elif 'news.sbs' in url:
        try:
            text = soup.find('div', {'itemprop': 'articleBody'})
            if text is None:
                text = soup.find('div', {'class':'text_area'})
            text = text.text
        except:
            return

    # KBS_NEWS
    elif 'news.kbs' in url:
        try:
            text = soup.find('div', {'id': 'cont_newstext'}).text
        except:
            return
    # JTBC_NEWS
    elif 'news.jtbc' in url:
        try:
            text = soup.find('title').nextSibling['content']
            text = re.sub(r'[&#39;|&quot;|&middot;]','',text)
        except:
            text = 'None'
    # YTN_NEWS

    elif 'ytn.co' in url:
        try:
            text = soup.find('div', {'class': 'paragraph'}).text
        except:
            return
    else:
        return

    text = text.replace('\n','').replace('\r','').replace('<br>','').replace('\t','')
    if len(text) > 1024:
        text = text[:1024]
    return text


def crawler(office_num, office_sec, media_cp, query, sort):
    title_text = []
    link_text = []
    date_text = []
    body=[]
    #
    page = 1
    # maxpage_t = (int(maxpage) - 1) * 10 + 1
    #
    # while page <= maxpage_t:
    url =("https://search.naver.com/search.naver?where=news&query="+query + "&sm=tab_opt"
          + "&sort="+str(sort)+ "&photo=0&field=0&pd=0&ds=&de=&docid=&related=0&mynews=1"
          + "&office_type=1"
          + "&office_section_code="+str(office_sec)
          + "&news_office_checked="+str(office_num)
          + "&nso=&is_sug_officeid=0&office_category=0&service_area=0")
    #
    response = requests.get(url)
    if response.status_code != 200:
        print("Load Error")
    #
    soup = BeautifulSoup(response.text, 'html.parser')

    # Extract title, link from <a>tag
    atags = soup.select('.news_tit')
    print("Extract NEWS Title and Link...")
    for atag in atags:
        title_text.append(atag.text)
        link_text.append(atag['href'])

    # Extract Date
    print("Extract NEWS Date...")
    date = soup.select('.info_group > span.info')
    for i, d in enumerate(date):
        title_text[i] += '('+ d.text + ')'



    # a,b,c,d,e = int(input("Enter five indices of article title"))
    for url in link_text:
        body.append(crawling_main_text(url))

    return title_text, body

    # date_text = [0]*len(body)
    # result = {'title': title_text,'article':body, 'link': link_text}
    # df = pd.DataFrame(result)
    #

    df.to_csv('data/result/{}_{}.tsv'.format(media_cp, query), sep='\t', index=False)

if __name__=='__main__':
    # maxpage = input("Enter the number of pages to crawl(int): ")
    query = input("Enter the search keyword:")
    # news_num = int(input("Enter the number of news to crawl(int):"))
    sort = int(input("Enter  the methods to crawl[Relevance:0/Latest:1/Oldest:2): "))
    media_cp = input("Enter the media company[KBS/MBC/SBS/JTBC/YTN/MAEIL/Yonhap]:")
    #
    # Set NEWS_OFFICE_NUM
    office_sec = 2
    if media_cp == 'KBS':
        office_num = 1056
    elif media_cp == 'JTBC':
        office_num = 1437
    elif media_cp == 'MAEIL':
        office_num = 1088
        office_sec = 6
    elif media_cp == 'MBC':
        office_num = 1214
    elif media_cp == 'SBS':
        office_num = 1055
    elif media_cp == 'YTN':
        office_num = 1052
    elif media_cp == 'Yonhap':
        office_num = 1001
    else:
        print("Invalid")

    pair, body = crawler(office_num=office_num,
                        office_sec=office_sec,
                        media_cp=media_cp,
                        query= query,
                        sort=sort)

    tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')

    summary = []
    for i in range(len(pair)):
        print(pair[i])
        text = body[i].replace('\n', '')
        input_ids = tokenizer.encode(text)
        #
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        #
        input_ids = torch.tensor([input_ids])
        summary_ids = model.generate(input_ids,
                                     eos_token_id=tokenizer.eos_token_id,
                                     max_length=512,
                                     length_penalty=2,
                                     num_beams=4, )
        output = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        summary.append(output)

    summary_emb = []

    for i in summary:
        summary_ids = tokenizer.encode(i)
        input_ids = [tokenizer.bos_token_id] + summary_ids + [tokenizer.eos_token_id]
        input_ids = np.array([input_ids])
        #
        input_pad = np.pad(input_ids[0], (0, 512 - len(input_ids[0])), 'constant',
                        constant_values = 0)
        input_pad = np.expand_dims(input_pad, axis=0)
        summary_emb.append(input_pad)
    matrix = torch.tensor(np.concatenate(summary_emb)).float()
    numerator = matrix.matmul(matrix.transpose(0,1))
    denominator = torch.norm(matrix) * torch.norm(matrix.transpose(0,1))
    cos_similarity = numerator/denominator
    #
    cos_similarity = torch.triu(cos_similarity)
    cos_similarity[np.where(cos_similarity == 0)] = 100
    # 4 --> change able parameter
    tmp=torch.sort(cos_similarity.view([1,-1]),descending=False)[0][:,:2]
    #
    torch.where(cos_similarity <= tmp[:,-1])
    top_n= set(list(torch.cat(torch.where(cos_similarity <= tmp[:, -1]), dim=0).numpy()))



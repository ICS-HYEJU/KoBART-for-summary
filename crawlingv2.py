from bs4 import BeautifulSoup
from datetime import datetime
import requests
import pandas as pd
import re

def crawling_main_text(url:str):
    response = requests.get(url)
    response.encoding = None
    html = response.text
    soup = BeautifulSoup(html, 'html.parser')

    # YONHAP_NEWS
    if ('://yna' in url) | ('app.yonhapnews' in url):
        main_article = soup.find('div', {'class': 'story-news article'})
        if main_article == None:
            main_article = soup.find('div', {'class': 'article-txt'})

        text = main_article.text

    # MBC_NEWS
    elif '//imnews.imbc' in url:
        text = soup.find('div', {'itemprop': 'articleBody'}).text

    # MAEIL_NEWS(Miracle)
    elif 'mirakle.mk' in url:
        text = soup.find('div', {'class': 'view_txt'}).text

    # MAEIL_NEWS
    elif 'mk.co' in url:
        text = soup.find('div', {'class': 'art_txt'}).text

    # SBS_NEWS
    elif 'news.sbs' in url:
        text = soup.find('div', {'itemprop': 'articleBody'}).text

    # KBS_NEWS
    elif 'news.kbs' in url:
        text = soup.find('div', {'id': 'cont_newstext'}).text

    # JTBC_NEWS
    elif 'news.jtbc' in url:
        text = soup.find('div', {'class': 'article_content'}).text

    else:
        text = None
        print('choose another media company')

    return text.replace('\n','').replace('\r','').replace('<br>','').replace('\t','')


def crawler(office_num, query, sort):
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
          + "&office_type=1&office_section_code=2"
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
    for atag in atags:
        title_text.append(atag.text)
        link_text.append(atag['href'])

    # Extract Date
    date = soup.select('.info_group > span.info')
    pattern = '\d{4}.\d{2}.\d{2}'
    r = re.compile(pattern)
    for d in date:
        date_text += r.findall(str(d))

    # a,b,c,d,e = int(input("Enter five indices of article title"))
    for url in link_text:
        body.append(crawling_main_text(url))

    result = {'title': title_text,'article':body, 'link': link_text, 'date': date_text}
    df = pd.DataFrame(result)
    df.to_csv('data/result/craw_v2_01.tsv', sep='\t', index=False)

if __name__=='__main__':
    # maxpage = input("Enter the number of pages to crawl(int): ")
    query = input("Enter the search keyword:")
    # news_num = int(input("Enter the number of news to crawl(int):"))
    sort = int(input("Enter  the methods to crawl[Relevance:0/Latest:1/Oldest:2): "))
    media_cp = input("Enter the media company[KBS/JTBC/MBC/SBS/YTN/Yonhap]:")
    # s_date = input("start date(2024.10.30):")
    # e_date = input("end date(2024.11.01):")
    #
    # Set NEWS_OFFICE_NUM
    if media_cp == 'KBS':
        office_num = 1056
    elif media_cp == 'JTBC':
        office_num = 1437
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

    crawler(office_num,query,sort)

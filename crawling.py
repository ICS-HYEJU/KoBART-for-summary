from bs4 import BeautifulSoup
from datetime import datetime
import requests
import pandas as pd
import re

title_text=[]
link_text=[]
source_text=[]
date_text=[]
contents_text=[]
result={}

news_titles=[]
news_contents=[]
news_dates=[]

RESULT_PATH = 'data/result'
now = datetime.now()


def date_cleansing(test):
    try:
        # past news
        pattern = '\d+.(\d+).(\d+).'

        r = re.compile(pattern)
        match = r.search(test).group(0)  # 2018.11.05.
        date_text.append(match)

    except AttributeError:
        # latest news
        pattern = '\w* (\d\w*)'

        r = re.compile(pattern)
        match = r.search(test).group(1)
        # print(match)
        date_text.append(match)
def contents_cleansing(contents):
    first_cleansing_contents = re.sub('<dl>.*?</a> </div> </dd> <dd>', '',
                                      str(contents)).strip()
    second_cleansing_contents = re.sub('<ul class="relation_lst">.*?</dd>', '',
                                       first_cleansing_contents).strip()
    third_cleansing_contents = re.sub('<.+?>', '', second_cleansing_contents).strip()
    contents_text.append(third_cleansing_contents)
    #print(contents_text)

def crawler(maxpage,query,sort,s_date,e_date):

    s_from = s_date.replace(".","")
    e_to = e_date.replace(".","")
    page = 1
    maxpage_t =(int(maxpage)-1)*10+1

    while page <= maxpage_t:
        url = "https://search.naver.com/search.naver?where=news&query=" + query + "&sort="+sort+"&ds=" + s_date + "&de=" + e_date + "&nso=so%3Ar%2Cp%3Afrom" + s_from + "to" + e_to + "%2Ca%3A&start=" + str(page)

        response = requests.get(url)
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')

        atags = soup.select('.news_tit')
        for atag in atags:
            title_text.append(atag.text)
            link_text.append(atag['href'])

        source_lists = soup.select('.info_group > .press')
        for source_list in source_lists:
            source_text.append(source_list.text)

        date_lists = soup.select('.info_group > span.info')
        for date_list in date_lists:
            # 1�� 3�� ���� ���� ����
            if date_list.text.find("��") == -1:
                date_text.append(date_list.text)

        contents_lists = soup.select('.news_dsc')
        for contents_list in contents_lists:
            contents_cleansing(contents_list)


        result= {"date" : date_text , "title":title_text ,  "source" : source_text ,"contents": contents_text ,"link":link_text }

        # print(contents_text)

        df = pd.DataFrame(result)
        page += 10

    for link in result['link']:
        # get html
        news = requests.get(link)
        news_html = BeautifulSoup(news.text, "html.parser")
        # get news title
        title = news_html.select_one("#ct > div.media_end_head.go_trans > div.media_end_head_title > h2")
        if title == None:
            title = news_html.select_one("#content > div.end_ct > div > h2")
        # get news body
        content = news_html.select("article#dic_area")
        if content == []:
            content = news_html.select("#articeBody")
        # crawling only text
        content = ''.join(str(content))
        # cleaning
        pattern1 = '<[^>]*>'
        title = re.sub(pattern=pattern1, repl='', string=str(title))
        content = re.sub(pattern=pattern1, repl='', string=content)
        pattern2 = """[\n\n\n\n\n// flash ������ �������� ���� ���� ����\nfunction _flash_removeCallback() {}"""
        content = content.replace(pattern2, '')

        news_titles.append(title)
        news_contents.append(content)

        try:
            html_date = news_html.select_one(
                "div#ct> div.media_end_head.go_trans > div.media_end_head_info.nv_notrans > div.media_end_head_info_datestamp > div > span")
            news_date = html_date.attrs['data-date-time']
        except AttributeError:
            news_date = news_html.select_one("#content > div.end_ct > div > div.article_info > span > em")
            news_date = re.sub(pattern=pattern1, repl='', string=str(news_date))
        # get news date
        news_dates.append(news_date)

    news = {'title':news_titles, 'article':news_contents, 'date':news_dates}
    news_df = pd.DataFrame(news)

    # outputFileName = '%s-%s-%s  %s�� %s�� %s�� merging.xlsx' % (now.year, now.month, now.day, now.hour, now.minute, now.second)
    df.to_csv('data/result/craw1.tsv', sep='\t', index=False)
    news_df.to_csv('data/result/craw1_news.tsv', sep='\t', index=False)

def main():

    maxpage = input("the number of pages you want to: ")
    query = input("query: ")
    sort = input("the methods you want to(relevant order=0  up-to-date order=1  old order=2): ")
    s_date = input("state date(2024.10.31):")  #2019.01.04
    e_date = input("end date(2024.11.01):")   #2019.01.05

    crawler(maxpage,query,sort,s_date,e_date)

main()
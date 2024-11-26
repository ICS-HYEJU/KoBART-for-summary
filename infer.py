import os
import io
import streamlit as st
import torch
import numpy as np

from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
import lightning as L
#
from model.bart import KoBARTGeneration
from config.config import get_config_dict


cfg = get_config_dict()
device = torch.device('cuda:{}'.format(1))
torch.cuda.set_device(1)
chp = False
def load_model():
    tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    if chp:
        chp_path = '/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/model_chp/news/epoch=00-val_loss=1.367.ckpt'
        model = KoBARTGeneration.load_from_checkpoint(path=chp_path, config=cfg, tok=tokenizer, device=device)
    else:
        model = KoBARTGeneration(config=cfg, tok=tokenizer)
    return tokenizer,model.to(device)

tokenizer, model = load_model()

#
st.set_page_config(page_title="TextSnack", page_icon=":cookie:")
st.title('Do you want to eat some snacks?')
st.header('TextSnack :cookie:')
# Sidebar
with st.sidebar:
    st.title('Snack on Info, Save Time!')
    st.header(':runner: Run')
    #
    menu = ['Upload File','Upload Article','Crawling Keyword','Recommend Article']
    choice = st.sidebar.selectbox('Menu', menu)


if choice == menu[0]:
    # Upload file
    txt_file = st.file_uploader('Upload your text', type=['txt', 'jsonl', 'json', 'tsv', 'csv'])
    out = st.empty()

    if txt_file:
        out.write("File uploaded successfully")
        txt = txt_file.getvalue().decode('utf-8')
        st.text_area("File content: ", txt , height=400)

        st.header('Summarization :pencil:')
        with st.spinner('processing..'):
            text = txt.replace('\n', '')
            input_ids = tokenizer.encode(text)
            #
            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            #
            input_ids = torch.tensor([input_ids])
            #
            model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization').to(device)
            #
            # chp_path = '/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/model_chp/news/epoch=02-val_loss=1.441.ckpt'
            # checkpoint = torch.load(chp_path, map_location=device)
            # new_state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
            # model.load_state_dict(new_state_dict)
            #
            summary_ids = model.generate(input_ids.to(device),
                                        eos_token_id=tokenizer.eos_token_id,
                                        max_length=512,
                                        length_penalty=2,
                                        num_beams=4,)
            output = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        st.write(output)

elif choice == menu[1]:
    # Text Input
    txt_input = st.text_area('Enter your text','', height=200)
    if txt_input:
        # st.header("Original Text")
        # st.write(txt_input)

        st.header('Summarization :pencil:')

        with st.spinner('processing..'):
            text = txt_input.replace('\n', '')
            input_ids = tokenizer.encode(text)
            #
            input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
            #
            input_ids = torch.tensor([input_ids])
            #
            model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization').to(device)
            #
            # chp_path = '/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/model_chp/news/epoch=02-val_loss=1.441.ckpt'
            # checkpoint = torch.load(chp_path, map_location=device)
            # new_state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
            # model.load_state_dict(new_state_dict)
            #
            summary_ids = model.generate(input_ids.to(device),
                                         eos_token_id=tokenizer.eos_token_id,
                                         max_length=512,
                                         length_penalty=2,
                                         num_beams=4, )
            output = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        st.write(output)

elif choice == menu[2]:
    from crawlingv2 import crawler

    query = st.text_input('Enter the search keyword: ')
    sort = ['Relevance', 'Latest', 'Oldest']
    my_sort = st.selectbox('Choose method of search', sort)
    media_cp = ['KBS', 'MBC', 'SBS', 'JTBC', 'YTN', 'MAEIL', 'Yonhap']
    my_media = st.selectbox('Choose an article', media_cp)
    #
    # Set NEWS_OFFICE_NUM
    if media_cp:
        office_sec = 2
        if my_media == media_cp[0]:   #KBS
            office_num = 1056
        elif my_media == media_cp[1]: #MBC
            office_num = 1214
        elif my_media == media_cp[2]: #SBS
            office_num = 1055
        elif my_media == media_cp[3]: #JTBC
            office_num = 1437
        elif my_media == media_cp[4]: #YTN
            office_num = 1052
        elif my_media == media_cp[5]: #MAEIL
            office_num = 1088
            office_sec = 6
        elif my_media == media_cp[6]: #YONHAP
            office_num = 1001


        model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization').to(device)
        pair, article = crawler(office_num=office_num,
                                     office_sec=office_sec,
                                     media_cp=media_cp,
                                     query=query,
                                     sort=sort)

        st.header('Title')
        my_choice = st.selectbox('Choose an article', pair)
        if my_choice:
            i = pair.index(my_choice)

            st.header('Article')
            st.text_area('content', article[i], height=200)

            st.header('Summarization :pencil:')

            with st.spinner('processing..'):
                text = article[i].replace('\n', '')
                input_ids = tokenizer.encode(text)
                #
                input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
                #
                input_ids = torch.tensor([input_ids])
                #

                #
                # chp_path = '/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/model_chp/news/epoch=02-val_loss=1.441.ckpt'
                # checkpoint = torch.load(chp_path, map_location=device)
                # new_state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
                # model.load_state_dict(new_state_dict)
                #
                summary_ids = model.generate(input_ids.to(device),
                                             eos_token_id=tokenizer.eos_token_id,
                                             max_length=512,
                                             length_penalty=2,
                                             num_beams=4, )
                output = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            st.write(output)

elif choice == menu[3]:
    from crawlingv2 import crawler

    query = st.text_input('Enter the search keyword: ')
    sort = ['Relevance', 'Latest', 'Oldest']
    my_sort = st.selectbox('Choose method of search', sort)
    media_cp = ['KBS', 'MBC', 'SBS', 'JTBC', 'YTN', 'MAEIL', 'Yonhap']
    my_media = st.selectbox('Choose an article', media_cp)
    num_view = st.number_input('Choose the number of article you want', 1,5)

    #
    # Set NEWS_OFFICE_NUM
    if media_cp:
        office_sec = 2
        if my_media == media_cp[0]:  # KBS
            office_num = 1056
        elif my_media == media_cp[1]:  # MBC
            office_num = 1214
        elif my_media == media_cp[2]:  # SBS
            office_num = 1055
        elif my_media == media_cp[3]:  # JTBC
            office_num = 1437
        elif my_media == media_cp[4]:  # YTN
            office_num = 1052
        elif my_media == media_cp[5]:  # MAEIL
            office_num = 1088
            office_sec = 6
        elif my_media == media_cp[6]:  # YONHAP
            office_num = 1001

        if st.button('Search'):
            pair, body = crawler(office_num=office_num,
                                 office_sec=office_sec,
                                 media_cp=media_cp,
                                 query=query,
                                 sort=sort)
            st.markdown('{}Articles crawled'.format(len(pair)))

            summary = []
            with st.spinner('processing..'):
                model = BartForConditionalGeneration.from_pretrained('EbanLee/kobart-summary-v3').to(device)
                tokenizer = PreTrainedTokenizerFast.from_pretrained('EbanLee/kobart-summary-v3')
                for i in range(len(pair)):
                    print(pair[i])
                    if body[i] is None:
                        continue
                    text = body[i].replace('\n', '')
                    input_ids = tokenizer.encode(text)
                    #
                    input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
                    #
                    import time
                    start = time.time()
                    input_ids = torch.tensor([input_ids])
                    summary_ids = model.generate(input_ids.to(device),
                                                 eos_token_id=tokenizer.eos_token_id,
                                                 max_length=512,
                                                 length_penalty=2,
                                                 num_beams=4, )
                    output = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
                    end = time.time()
                    # st.header(f"{end - start:.5f} sec")
                    summary.append(output)


                summary_emb = []

                for i in summary:
                    summary_ids = tokenizer.encode(i)
                    input_ids = [tokenizer.bos_token_id] + summary_ids + [tokenizer.eos_token_id]
                    input_ids = np.array([input_ids])
                    #
                    input_pad = np.pad(input_ids[0], (0, 512 - len(input_ids[0])), 'constant',
                                       constant_values=0)
                    input_pad = np.expand_dims(input_pad, axis=0)
                    summary_emb.append(input_pad)

                if len(summary_emb) > 0:
                    matrix = torch.tensor(np.concatenate(summary_emb)).float().to(device)
                    numerator = matrix.matmul(matrix.transpose(0, 1))
                    denominator = torch.norm(matrix) * torch.norm(matrix.transpose(0, 1))
                    cos_similarity = numerator / denominator
                    #
                    cos_similarity = torch.triu(cos_similarity)
                    cos_similarity[torch.where(cos_similarity == 0)] = 100
                    # 4 --> change able parameter
                    tmp = torch.sort(cos_similarity.view([1, -1]), descending=False)[0][:, :num_view-1]
                    #
                    torch.where(cos_similarity <= tmp[:, -1])
                    top_n = set(list(torch.cat(torch.where(cos_similarity <= tmp[:, -1]), dim=0).cpu().numpy()))
                    #
                    cnt=1
                    # st.header('The result of search')
                    for i in top_n:
                        st.header('', divider='orange')
                        st.header(':pushpin: Title')
                        st.write(pair[i])
                        st.header(':newspaper: Article')
                        st.write(body[i])
                        st.header(':pencil: Summarization ')
                        st.write(summary[i])
                        cnt+=1
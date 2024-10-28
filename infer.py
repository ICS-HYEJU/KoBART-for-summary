import streamlit as st
import torch

from transformers import BartForConditionalGeneration
from transformers import PreTrainedTokenizerFast
import lightning as L
#
from model.bart import KoBARTGeneration
from config.config import get_config_dict


cfg = get_config_dict()
device = torch.device('cuda:{}'.format(1))
torch.cuda.set_device(1)
chp = True

#
st.set_page_config(page_title="TextSnack", page_icon=":cookie:")
st.title('Do you want to eat some snacks?')
st.header('TextSnack :cookie:')

# Sidebar
with st.sidebar:
    st.title('Snack on Info, Save Time!')
    st.header(':runner: Run')
    #
    menu = ['upload file']
    choice = st.sidebar.selectbox('Menu', menu)

# Upload file
txt_file = st.file_uploader('Upload your text', type=['txt','jsonl','json','tsv'])
out = st.empty()

# Load Tokenizer
tokenizer = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')

if txt_file:
    out.write("File uploaded successfully")
    txt = txt_file.getvalue().decode('utf-8')
    st.text_area("File content: ", txt, height=400)

    st.markdown('Summarization :pencil:')
    with st.spinner('processing..'):
        text = txt.replace('\n', '')
        input_ids = tokenizer.encode(text)
        #
        input_ids = [tokenizer.bos_token_id] + input_ids + [tokenizer.eos_token_id]
        #
        input_ids = torch.tensor([input_ids])
        #
        model = BartForConditionalGeneration.from_pretrained('digit82/kobart-summarization')
        #
        chp_path = '/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/model_chp/news/epoch=00-val_loss=1.367.ckpt'
        checkpoint = torch.load(chp_path, map_location=device)
        new_state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(new_state_dict)
        #
        summary_ids = model.generate(input_ids,
                                    eos_token_id=tokenizer.eos_token_id,
                                    max_length=512,
                                    length_penalty=2,
                                    num_beams=8,)
        output = tokenizer.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
    st.write(output)

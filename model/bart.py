import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration
import lightning as L
import time
from lightning.pytorch.loggers import TensorBoardLogger

class KoBARTGeneration(L.LightningModule):
    def __init__(self, config, tok):
        super().__init__()
        #
        self.cfg = config
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = tok.pad_token_id
        self.tok = tok
        #
        self.pretrained_path = 'digit82/kobart-summarization'
        self.model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=self.pretrained_path)
        #
        self.outputs = []

    def load_from_checkpoint(self, path ,device):
        checkpoint = torch.load(path, map_location=device)
        new_state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
        self.model.load_state_dict(new_state_dict)

    def forward(self,inputs):
        attn_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attn_mask = inputs['dec_input_ids'].ne(self.pad_token_id).float()

        return self.model(input_ids=inputs['input_ids'].to(self.device),
                          attention_mask=attn_mask.to(self.device),
                          decoder_input_ids=inputs['dec_input_ids'].to(self.device),
                          decoder_attention_mask=decoder_attn_mask.to(self.device),
                          labels=inputs['label_ids'].to(self.device),
                          return_dict=True)

    def training_step(self, batch, batch_idx):
        outs = self.forward(batch)
        loss = outs.loss
        self.log('train_loss', loss, on_step=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outs = self(batch)
        loss = outs['loss']
        self.log('val_loss', loss, on_step=True, prog_bar=True)
        self.outputs.append({"val_loss": loss})

    def on_validation_epoch_end(self):
        loss = torch.stack([x["val_loss"] for x in self.outputs]).mean()
        self.log("val_epoch_loss", loss, prog_bar=True)
        self.outputs.clear()

    def configure_optimizers(self):
        # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.cfg.solver['lr'], correct_bias=False)
        #
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.trainer.estimated_stepping_batches * 0.1),
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        lr_scheduler = {'scheduler': scheduler,
                        'monitor': self.cfg.scheduler['monitor'],
                        'interval': self.cfg.scheduler['interval'],     # called after each training step
                        'frequency': self.cfg.scheduler['frequency']}

        return [optimizer], [lr_scheduler]

    def inference(self, input_text):
        # print("KoBART Summary Test")
        # print("## origin NEWS data: ")
        # print(input_text)
        # assert input_text is not None, f'input text is None'


        text = input_text.replace('\n', '')
        raw_input_ids = self.tok.encode(text)
        #
        input_ids = [self.tok.bos_token_id] + raw_input_ids + [self.tok.eos_token_id]
        #
        input_ids = torch.tensor([input_ids]).to(self.device)
        summary_ids = self.model.generate(input_ids,
                                     eos_token_id=self.tok.eos_token_id,
                                     max_length=512,
                                     num_beams=4)
        output = self.tok.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        # print("## Model Summary: ")
        # print(output)
        return output


if __name__ == '__main__':
    from collections import defaultdict
    import pandas as pd
    import glob
    from rouge import Rouge
    from transformers import PreTrainedTokenizerFast
    from config.config import get_config_dict

    #
    cfg = get_config_dict()
    tok = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    rouge = Rouge()
    #
    if cfg.device['gpu_id'] is not None:
        device = torch.device('cuda:{}'.format(cfg.device['gpu_id']))
        torch.cuda.set_device(cfg.device['gpu_id'])
    else:
        device = torch.device('cpu')
    #
    start = time.time()
    path = sorted(glob.glob('/storage/hjchoi/model_save/news/*.ckpt'), reverse=True)

    score_dict = {}
    #
    test_path = '/storage/hjchoi/data_dacon/test.tsv'
    train_path = '/storage/hjchoi/data_dacon/train.tsv'
    line = pd.read_csv(test_path, sep="\t")
    cnt = len(line)
    threshold=0
    qt_data =[]
    for ckpt in path:
        if ckpt == '/storage/hjchoi/model_save/news/epoch=49-val_loss=1.570.ckpt':
            continue
    # ckpt = '/storage/hjchoi/dacon_chp/last.ckpt'
        model = KoBARTGeneration(config=cfg, tok=tok)
        model.load_from_checkpoint(path=ckpt,device=device)
        model.to(device)
    #
        result = defaultdict(lambda: {'r': 0, 'p': 0, 'f': 0})
        for i in range(len(line)):
            threshold=0
            input_text = line.iloc[i]['news']
            reference = line.iloc[i]['summary']
            if len(input_text) > 1024:
                cnt -= 1
                continue
            #
            model_output = model.inference(input_text)
            #
            score = rouge.get_scores(model_output, reference, avg=True)
            for metric, values in score.items():
                if values['f'] != 0.0:
                    for key in values:
                        result[metric][key] += values[key]
                else:
                    cnt -= 1
            print(i, score)
        print(qt_data)
        result = dict(result)

        for m, v in result.items():
            for k in v:
                result[m][k] /= cnt
        # score_dict[str(ckpt.split('/')[-1])] = result
        df = pd.DataFrame.from_dict(result)
        print(df)
        end=time.time()
        print(f'{end-start}s')
    # df.to_csv('/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/model_chp/Dacon_.csv')
    # view = True
    #
    # if view:
    #     pd.set_option('display.max_columns', None)
    #     pd.set_option('display.max_rows', None)
    #     df = pd.read_csv('/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/model_chp/news/rouge.csv')
    #     print(df)
    # else:
    #     if chp:
    #         for ckpt in path:
    #             model = KoBARTGeneration.load_from_checkpoint(path=ckpt, config=cfg, tok=tok, mode='fit',device=device)
    #             model_output = model.inference(input_text)
    #             #
    #             score = rouge.get_scores(model_output, reference, avg=True)
    #             score_dict[str(ckpt.split('/')[-1])] = score
    #         df = pd.DataFrame.from_dict(score_dict)
    #         print(df)
    #         df.to_csv('/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/model_chp/news/rouge.csv',header=False)
    #     else:
    #         model = KoBARTGeneration(config=cfg,tok=tok, mode='fit')
    #         model_output = model.inference(input_text)
    #     #


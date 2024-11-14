import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration
import lightning as L
from lightning.pytorch.loggers import TensorBoardLogger

class KoBARTGeneration(L.LightningModule):
    def __init__(self, config, tok, mode):
        super().__init__()
        #
        self.cfg = config
        self.bos_token = '<s>'
        self.eos_token = '</s>'
        self.pad_token_id = tok.pad_token_id
        self.tok = tok
        self.mode = mode
        #
        self.pretrained_path = 'digit82/kobart-summarization'
        self.model = BartForConditionalGeneration.from_pretrained(pretrained_model_name_or_path=self.pretrained_path)
        if self.mode == 'fit':
            self.model.train()
        else:
            self.model.eval()
        #
        self.outputs = []

    @classmethod
    def load_from_checkpoint(cls, path, config, tok, mode,device):
        model = BartForConditionalGeneration.from_pretrained(config.dataset_info['pretrained_name'])
        checkpoint = torch.load(path, map_location=device)
        new_state_dict = {k.replace('model.', '', 1): v for k, v in checkpoint['state_dict'].items()}
        model.load_state_dict(new_state_dict)
        return cls(model, tok, mode)

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
        print("KoBART Summary Test")
        print("## origin NEWS data: ")
        print(input_text)
        assert input_text is not None, f'input text is None'


        text = input_text.replace('\n', '')
        raw_input_ids = self.tok.encode(text)
        #
        input_ids = [self.tok.bos_token_id] + raw_input_ids + [self.tok.eos_token_id]
        #
        input_ids = torch.tensor([input_ids])
        summary_ids = self.model.generate(input_ids,
                                     eos_token_id=self.tok.eos_token_id,
                                     max_length=512,
                                     num_beams=4)
        output = self.tok.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
        print("## Model Summary: ")
        print(output)
        return output


if __name__ == '__main__':
    import pandas as pd
    from transformers import PreTrainedTokenizerFast
    from config.config import get_config_dict
    cfg = get_config_dict()
    tok = PreTrainedTokenizerFast.from_pretrained('digit82/kobart-summarization')
    # tok = PreTrainedTokenizerFast.from_pretrained('ainize/kobert-news')
    if cfg.device['gpu_id'] is not None:
        device = torch.device('cuda:{}'.format(cfg.device['gpu_id']))
        torch.cuda.set_device(cfg.device['gpu_id'])
    else:
        device = torch.device('cpu')
    #
    test_path = '/storage/hjchoi/Document_Summary_text/Validation/news.tsv'
    line = pd.read_csv(test_path, sep="\t")
    input_text = line['text'][0]
    reference = line['summary'][0]
    #
    chp = False
    if chp:
        chp_path = '/home/hjchoi/PycharmProjects/KoBART-for-summary/checkpoint/model_chp/news/epoch=00-val_loss=1.367.ckpt'
        model = KoBARTGeneration.load_from_checkpoint(path=chp_path, config=cfg, tok=tok, mode='fit')
        model_output = model.inference(input_text)
    else:
        model = KoBARTGeneration(config=cfg,tok=tok, mode='fit')
        model_output = model.inference(input_text)
    #
    from rouge import Rouge
    rouge = Rouge()
    rouge.get_scores(model_output, reference, avg=True)
    print(rouge.get_scores(model_output, reference, avg=True))


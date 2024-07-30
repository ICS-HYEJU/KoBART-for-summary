import torch
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from transformers import BartForConditionalGeneration
import lightning as L


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
        self.model = BartForConditionalGeneration.from_pretrained(config.dataset_info['pretrained_name'])
        self.model.train()
        #
        self.outputs = []

    def forward(self,inputs):
        attn_mask = inputs['input_ids'].ne(self.pad_token_id).float()
        decoder_attn_mask = inputs['dec_input_ids'].ne(self.pad_token_id).float()

        return self.model(input_ids=inputs['input_ids'].to(self.device),
                          attention_mask=attn_mask.to(self.device),
                          decoder_input_ids=inputs['dec_input_ids'].to(self.device),
                          decoder_attention_mask=decoder_attn_mask.to(self.device),
                          labels=inputs['label_ids'].to(self.device),
                          return_dict=True)

    def training_step(self, batch,batch_idx):
        outs = self.forward(batch)
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outs = self(batch)
        loss = outs['loss']
        self.outputs.append({"loss": loss})

    def on_validation_epoch_end(self):
        loss = torch.stack([x["loss"] for x in self.outputs]).mean()
        self.log("val_loss", loss, prog_bar=True)
        self.outputs.clear()
        return {'avg_val_loss': loss}

    def inference(self, input_text):
        print("KoBART Summary Test")
        print("## origin NEWS data: ")
        print(input_text)

        if input_text:
            text = input_text.replace('\n', '')
            raw_input_ids = self.tok.encode(text)
            #
            input_idx = [self.tok.bos_token_id] + raw_input_ids + [self.tok.eos_token_id]
            #
            input_ids = torch.tensor([input_idx])
            summary_ids = self.model.generate(input_ids,
                                              eos_token_id=self.tok.eos_token_id,
                                              max_length=self.config.max_len,
                                              num_beams=4)
            output = self.tok.decode(summary_ids.squeeze().tolist(), skip_special_tokens=True)
            print(output)

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
                        'interval': self.cfg.scheduler['interval'],
                        'frequency': self.cfg.scheduler['frequency']}

        return [optimizer], [lr_scheduler]


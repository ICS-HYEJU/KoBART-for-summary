import torch
import os

from transformers import PreTrainedTokenizerFast

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint


class Engine():
    def __init__(self, cfg, mode:str, device):
        self.cfg = cfg
        self.mode = mode
        self.device = device

        # ===== Save Path =====
        self.save_path = self.make_save_path()

        # ===== DataLoader =====
        self.dataloader = self.get_dataloader()

        # ===== Model =====
        self.model = self.build_model()

        # ===== Make CheckPoint =====
        self.checkpoint = self.build_checkpoint()

    def build_model(self):
        name = self.cfg.model['name']
        if name == 'BART':
            from model.bart import KoBARTGeneration
            tok = PreTrainedTokenizerFast.from_pretrained(self.cfg.dataset_info['pretrained_name'])
            model = KoBARTGeneration(config=self.cfg, tok=tok)
        else:
            raise NotImplementedError(f"The required model is not implemented yet...")
        return model.to(self.device)

    def get_dataloader(self):
        from data.dataloader import KobartSummaryDataModule
        datamodule = KobartSummaryDataModule(
            train_file=self.cfg.dataset_info['train_file'],
            test_file=self.cfg.dataset_info['test_file'],
            max_len=self.cfg.dataset_info['max_len'],
            batch_size=self.cfg.dataset_info['batch_size'],
            num_workers=self.cfg.dataset_info['num_workers'],
            pretrained_name=self.cfg.dataset_info['pretrained_name']
        )
        if self.mode == 'fit':
            datamodule.setup('fit')
        elif self.mode == 'test':
            datamodule.setup('test')
        elif self.mode == 'predict':
            datamodule.setup('predict')
        else:
            raise ValueError(f'mode {self.mode} is invalid...[train \| test \| predict]')

        return datamodule

    def make_save_path(self):
        save_pretrain = os.path.join(self.cfg.path['save_base_path'],
                                     self.cfg.model['name'] + "_pretrain")
        os.makedirs(save_pretrain, exist_ok=True)
        return save_pretrain

    def build_checkpoint(self):
        checkpoint_callback = ModelCheckpoint(
            monitor='train_loss',
            dirpath=self.cfg.weight_info['checkpoint'],
            filename=self.cfg.weight_info['save_fname'],
            mode=self.cfg.weight_info['mode'],
            verbose=True,
            save_last=True,
            save_top_k=-1)
        return checkpoint_callback

    def start_train(self):
        trainer = L.Trainer(max_epochs=self.cfg.dataset_info['max_epoch'],
                            accelerator=self.cfg.scheduler['accelerator'],
                            devices=[self.cfg.device['gpu_id']],
                            gradient_clip_val=self.cfg.solver['gradient_clip_val'],
                            callbacks=[self.checkpoint],
                            accumulate_grad_batches=3,
                            )
        if self.mode=='fit':
            trainer.fit(self.model, self.dataloader)
        elif self.mode=='test':
            trainer.test(self.model, self.dataloader)
        elif self.mode=='predict':
            trainer.predict(self.model, self.dataloader)


if __name__ == '__main__':
    torch.set_float32_matmul_precision('high')

    from config.config import get_config_dict
    cfg = get_config_dict()
    #
    if cfg.device['gpu_id'] is not None:
        device = torch.device('cuda:{}'.format(cfg.device['gpu_id']))
        torch.cuda.set_device(cfg.device['gpu_id'])
    else:
        device = torch.device('cpu')
    #
    engine = Engine(cfg, mode='fit', device=device)
    engine.start_train()
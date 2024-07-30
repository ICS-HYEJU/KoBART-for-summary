import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader

from transformers import PreTrainedTokenizerFast
from transformers.optimization import AdamW, get_linear_schedule_with_warmup

from collections import defaultdict

import os
import glob
import ast
from tqdm import tqdm, trange
from functools import partial

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint

from collections import defaultdict

import streamlit as st

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
        if self.mode=='fit':
            datamodule.setup('fit')
        elif self.mode=='test':
            datamodule.setup('test')
        elif self.mode=='predict':
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
            monitor=self.cfg.scheduler['monitor'],
            dirpath=self.cfg.weight_info['checkpoint'],
            filename=self.cfg.weight_info['save_fname'],
            verbose=True,
            save_last=True,
            mode=self.cfg.scheduler['mode'],
            save_top_k=3)
        return checkpoint_callback

    def start_train(self):
        trainer = L.Trainer(max_epochs=self.cfg.dataset_info['max_epoch'],
                            accelerator=self.cfg.scheduler['accelerator'],
                            devices=[self.cfg.device['gpu_id']],
                            gradient_clip_val=self.cfg.solver['gradient_clip_val'],
                            callbacks=[self.checkpoint],
                            accumulate_grad_batches=5
                            )
        if self.mode=='fit':
            trainer.fit(self.model, self.dataloader)
        elif self.mode=='test':
            trainer.test(self.model, self.dataloader)
        elif self.mode=='predict':
            trainer.predict(self.model, self.dataloader)
    # def one_epoch(self):
    #     for batch_idx, data in enumerate(self.dataloader.train_dataloader()):
    #         data_d = dict()
    #         data_d['input_ids'] = data['input_ids'].to(self.device)
    #         data_d['dec_input_ids'] = data['dec_input_ids'].to(self.device)
    #         data_d['label_ids'] = data['label_ids'].to(self.device)

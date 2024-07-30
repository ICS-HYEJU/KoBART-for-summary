import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

import lightning as L
from transformers import PreTrainedTokenizerFast
from data.dataset import KoBARTSummaryDataset

class KobartSummaryDataModule(L.LightningDataModule):
    def __init__(self, train_file, test_file, max_len=512, batch_size=8, num_workers=0,
                 pretrained_name='gogamza/kobart-base-v1'):
        super().__init__()
        #
        self.bs = batch_size
        self.max_len = max_len
        self.num_workers = num_workers
        #
        self.train_file_path = train_file
        self.test_file_path = test_file
        self.tok = PreTrainedTokenizerFast.from_pretrained(pretrained_name)

    def setup(self, stage:str):
        if stage == 'fit':
            self.train = KoBARTSummaryDataset(self.train_file_path,
                                              max_len=self.max_len,
                                              tokenizer=self.tok)
            self.val = KoBARTSummaryDataset(self.test_file_path,
                                            max_len=self.max_len,
                                            tokenizer=self.tok)
        elif stage == 'test':
            self.test = KoBARTSummaryDataset(self.test_file_path,
                                            max_len=self.max_len,
                                            tokenizer=self.tok)
        elif stage == 'predict':
            self.predict = KoBARTSummaryDataset(self.test_file_path,
                                             max_len=self.max_len,
                                             tokenizer=self.tok)
        else:
            raise ValueError(f'stage {stage} is invalid...[train \| test \| predict]')

    def train_dataloader(self):
        train = DataLoader(self.train, batch_size=self.bs, num_workers=self.num_workers)
        return train

    def val_dataloader(self):
        val = DataLoader(self.val, batch_size=self.bs, num_workers=self.num_workers)
        return val

    def test_dataloader(self):
        test = DataLoader(self.test, batch_size=self.bs, num_workers=self.num_workers)
        return test

    def predict_dataloader(self):
        pred = DataLoader(self.predict, batch_size=self.bs, num_workers=self.num_workers)
        return pred

if __name__ == '__main__':
    from config.config import get_config_dict
    cfg = get_config_dict()
    dataloader = KobartSummaryDataModule(
        train_file=cfg.dataset_info['train_file'],
        test_file=cfg.dataset_info['test_file'],
        max_len =cfg.dataset_info['max_len'],
        batch_size=cfg.dataset_info['batch_size'],
        num_workers=0,
        pretrained_name='gogamza/kobart-base-v1'
    )
    dataloader.setup('fit')
    for batch_idx, data in enumerate(dataloader.train_dataloader()):
        print(batch_idx, data)
        break

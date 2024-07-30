import numpy as np
import pandas as pd

import lightning as L


class KoBARTSummaryDataset(L.LightningDataModule):
    def __init__(self, file, tokenizer, max_len, ignore_index = -100):
        super().__init__()
        #
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.docs = pd.read_csv(file, sep="\t")
        self.data_len = self.docs.shape[0]
        self.pad_idx = self.tokenizer.pad_token_id
        self.ignore_index = ignore_index

    def __len__(self):
        return self.data_len

    def __getitem__(self, idx):
        instance = self.docs.iloc[idx]

        # input_ids = Integer(data['news']) + Padding
        input_ids = self.tokenizer.encode(instance['news'])
        input_ids = self.add_padding_data(input_ids)

        # label_ids = Integer(data['summary']) + [EOS] token
        label_ids = self.tokenizer.encode(instance['summary'])
        label_ids.append(self.tokenizer.eos_token_id)

        # dec_input : [EOS] + label_id(except EOS token) + Padding
        dec_input_ids = [self.tokenizer.eos_token_id]
        dec_input_ids += label_ids[:-1]
        dec_input_ids = self.add_padding_data(dec_input_ids)

        # add noise token to labe_ids
        label_ids = self.add_ignored_data(label_ids)

        return {'input_ids': np.array(input_ids,dtype=np.int_),
                'dec_input_ids': np.array(dec_input_ids, dtype=np.int_),
                'label_ids': np.array(label_ids,dtype=np.int_)}

    # add_pad = [ 'A', 'B', 'C', 'D', 'E', 'F', ...'PAD', 'PAD', 'PAD']
    # len(add_pad) = self.max_len
    def add_padding_data(self, inputs):
        if len(inputs) < self.max_len:
            pad = np.array([self.pad_idx] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, pad])
        else:
            inputs = inputs[:self.max_len]
        return inputs

    # BART uses Denoising --> this function adds noising to origin data
    # add_ignore = [ 'A', 'B', 'C', 'D', 'E', 'F', ... -100, -100, -100]
    # len(add_ignore) = self.max_len
    def add_ignored_data(self,inputs):
        if len(inputs) < self.max_len:
            ignore = np.array([self.ignore_index] * (self.max_len - len(inputs)))
            inputs = np.concatenate([inputs, ignore])
        else:
            inputs = inputs[:self.max_len]
        return inputs



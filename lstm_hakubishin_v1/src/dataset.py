import torch
import pandas as pd
import numpy as np
from keras.preprocessing import sequence


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, is_train: bool = True) -> None:
        super().__init__
        self.is_train = is_train
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int):
        city_ids = self.df["city_id"].values[index]
        # Remove last city_id for input sequence
        city_id_tensor = city_ids[:-1]
        # Remove first city_id for target sequence
        target_tensor = city_ids[1:]

        if self.is_train:
            return (city_id_tensor, target_tensor)
        else:
            return (city_id_tensor,)


class Collator(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        seqs = [item[0] for item in batch]
        if self.is_train:
            targets = [item[1] for item in batch]

        lens = [len(s) for s in seqs]
        max_len = max(lens)
        seqs = sequence.pad_sequences(seqs, maxlen=max_len)
        seqs = torch.tensor(seqs, dtype=torch.long)
        if self.is_train:
            targets = sequence.pad_sequences(targets, maxlen=max_len)
            targets = torch.tensor(targets, dtype=torch.long)
            return (seqs, targets)

        return (seqs,)

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
        # Remove last city_id for input sequence
        city_id_tensor = self.df["past_city_id"].values[index]
        booker_country_tensor = self.df["booker_country"].values[index]

        # Remove first city_id for target sequence
        target_tensor = self.df["city_id"].values[index]

        if self.is_train:
            return (city_id_tensor, booker_country_tensor, target_tensor)
        else:
            return (city_id_tensor, booker_country_tensor)


class Collator(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        seqs = [item[0] for item in batch]
        cats = [item[1] for item in batch]
        if self.is_train:
            targets = [item[2] for item in batch]

        def _pad_sequences(data, maxlen: int) -> torch.tensor:
            data = sequence.pad_sequences(data, maxlen=maxlen)
            return torch.tensor(data, dtype=torch.long)

        lens = [len(s) for s in seqs]
        seqs = _pad_sequences(seqs, max(lens))
        cats = _pad_sequences(cats, max(lens))
        if self.is_train:
            targets = _pad_sequences(targets, max(lens))
            return (seqs, cats, targets)

        return (seqs, cats)

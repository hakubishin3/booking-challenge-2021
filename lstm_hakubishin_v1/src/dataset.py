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
        city_id_tensor = city_ids[:-1]
        target_tensor = city_ids[1:]

        if self.is_train:
            return (city_id_tensor, target_tensor)
        else:
            return (city_id_tensor,)


def collate_fn(batch):
    seqs = [item[0] for item in batch]
    target = [item[1] for item in batch]
    lens = [len(x) for x in seqs]
    max_len = max(lens)
    seqs = sequence.pad_sequences(seqs, maxlen=int(max_len))
    seqs = torch.tensor(seqs, dtype=torch.long)
    target = sequence.pad_sequences(target, maxlen=int(max_len))
    target = torch.tensor(target, dtype=torch.long)
    return (
        seqs,
        target,
    )

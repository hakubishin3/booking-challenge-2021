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
        city_id_tensor = self.df["past_city_id"].values[index]
        booker_country_tensor = self.df["booker_country"].values[index]
        device_class_tensor = self.df["device_class"].values[index]
        affiliate_id_tensor = self.df["affiliate_id"].values[index]
        target_tensor = self.df["city_id"].values[index]

        if self.is_train:
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                target_tensor,
            )
        else:
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
            )


class Collator(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        city_id_tensor = [item[0] for item in batch]
        booker_country_tensor = [item[1] for item in batch]
        device_class_tensor = [item[2] for item in batch]
        affiliate_id_tensor = [item[3] for item in batch]
        if self.is_train:
            targets = [item[-1] for item in batch]

        def _pad_sequences(data, maxlen: int) -> torch.tensor:
            data = sequence.pad_sequences(data, maxlen=maxlen)
            return torch.tensor(data, dtype=torch.long)

        lens = [len(s) for s in city_id_tensor]
        city_id_tensor = _pad_sequences(city_id_tensor, max(lens))
        booker_country_tensor = _pad_sequences(booker_country_tensor, max(lens))
        device_class_tensor = _pad_sequences(device_class_tensor, max(lens))
        affiliate_id_tensor = _pad_sequences(affiliate_id_tensor, max(lens))
        if self.is_train:
            targets = _pad_sequences(targets, max(lens))
            return (
                city_id_tensor,
                booker_country_tensor,
                device_class_tensor,
                affiliate_id_tensor,
                targets,
            )

        return (
            city_id_tensor,
            booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
        )

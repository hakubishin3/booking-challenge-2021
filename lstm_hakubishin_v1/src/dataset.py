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
        target_tensor = self.df["city_id"].values[index][
            -1
        ]  # extrast last value of sequence
        month_checkin_tensor = self.df["month_checkin"].values[index]
        num_checkin_tensor = self.df["num_checkin"].values[index]
        days_stay_tensor = self.df["days_stay"].values[index]
        days_move_tensor = self.df["days_move"].values[index]
        hotel_country_tensor = self.df["past_hotel_country"].values[index]
        num_visit_drop_duplicates_tensor = self.df["num_visit_drop_duplicates"].values[index]
        num_visit_tensor = self.df["num_visit"].values[index]
        num_visit_same_city_tensor = self.df["num_visit_same_city"].values[index]
        num_stay_consecutively_tensor = self.df["num_stay_consecutively"].values[index]
        city_embedding_tensor = np.asarray(self.df["city_embedding"].values[index])
        num_past_city_to_current_city_embedding_tensor = np.asarray(self.df["num_past_city_to_embedding"].values[index])

        input_tensors = (
            city_id_tensor,
            booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
            month_checkin_tensor,
            num_checkin_tensor,
            days_stay_tensor,
            days_move_tensor,
            hotel_country_tensor,
            num_visit_drop_duplicates_tensor,
            num_visit_tensor,
            num_visit_same_city_tensor,
            num_stay_consecutively_tensor,
            city_embedding_tensor,
            num_past_city_to_current_city_embedding_tensor,
        )
        target_tensors = (
            target_tensor,
        )
        if self.is_train:
            return input_tensors + target_tensors
        else:
            return input_tensors


class Collator(object):
    def __init__(self, is_train=True):
        self.is_train = is_train

    def __call__(self, batch):
        city_id_tensor = [item[0] for item in batch]
        booker_country_tensor = [item[1] for item in batch]
        device_class_tensor = [item[2] for item in batch]
        affiliate_id_tensor = [item[3] for item in batch]
        month_checkin_tensor = [item[4] for item in batch]
        num_checkin_tensor = [item[5] for item in batch]
        days_stay_tensor = [item[6] for item in batch]
        days_move_tensor = [item[7] for item in batch]
        hotel_country_tensor = [item[8] for item in batch]
        num_visit_drop_duplicates_tensor = [item[9] for item in batch]
        num_visit_tensor = [item[10] for item in batch]
        num_visit_same_city_tensor = [item[11] for item in batch]
        num_stay_consecutively_tensor = [item[12] for item in batch]
        city_embedding_tensor = [item[13] for item in batch]
        num_past_city_to_current_city_embedding_tensor = [item[14] for item in batch]
        if self.is_train:
            targets = [item[-1] for item in batch]

        def _pad_sequences(data, maxlen: int, dtype=torch.long) -> torch.tensor:
            data = sequence.pad_sequences(data, maxlen=maxlen)
            return torch.tensor(data, dtype=dtype)

        lens = [len(s) for s in city_id_tensor]
        city_id_tensor = _pad_sequences(city_id_tensor, max(lens))
        booker_country_tensor = _pad_sequences(booker_country_tensor, max(lens))
        device_class_tensor = _pad_sequences(device_class_tensor, max(lens))
        affiliate_id_tensor = _pad_sequences(affiliate_id_tensor, max(lens))
        month_checkin_tensor = _pad_sequences(month_checkin_tensor, max(lens))
        num_checkin_tensor = _pad_sequences(
            num_checkin_tensor, max(lens), dtype=torch.float
        )
        days_stay_tensor = _pad_sequences(
            days_stay_tensor, max(lens), dtype=torch.float
        )
        days_move_tensor = _pad_sequences(
            days_move_tensor, max(lens), dtype=torch.float
        )
        hotel_country_tensor = _pad_sequences(hotel_country_tensor, max(lens))
        num_visit_drop_duplicates_tensor = _pad_sequences(
            num_visit_drop_duplicates_tensor, max(lens), dtype=torch.float
        )
        num_visit_tensor = _pad_sequences(
            num_visit_tensor, max(lens), dtype=torch.float
        )
        num_visit_same_city_tensor = _pad_sequences(
            num_visit_same_city_tensor, max(lens), dtype=torch.float
        )
        num_stay_consecutively_tensor = _pad_sequences(
            num_stay_consecutively_tensor, max(lens), dtype=torch.float
        )
        city_embedding_tensor = _pad_sequences(
            city_embedding_tensor, max(lens), dtype=torch.float
        )
        num_past_city_to_current_city_embedding_tensor = _pad_sequences(
            num_past_city_to_current_city_embedding_tensor, max(lens), dtype=torch.float
        )

        input_tensors = (
            city_id_tensor,
            booker_country_tensor,
            device_class_tensor,
            affiliate_id_tensor,
            month_checkin_tensor,
            num_checkin_tensor,
            days_stay_tensor,
            days_move_tensor,
            hotel_country_tensor,
            num_visit_drop_duplicates_tensor,
            num_visit_tensor,
            num_visit_same_city_tensor,
            num_stay_consecutively_tensor,
            city_embedding_tensor,
            num_past_city_to_current_city_embedding_tensor,
        )
        if self.is_train:
            targets = torch.tensor(targets, dtype=torch.long)
            return input_tensors + (targets,)

        return input_tensors

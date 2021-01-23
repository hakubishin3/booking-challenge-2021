import os
import sys
import time
import pathlib
import pandas as pd
from base import Feature
from contextlib import contextmanager
#from encoding_functions import label_encoding

FE_DIR = "./data/features/"
PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))
from src import load_train_test_set, preprocess_train_test_set


CATEGORICAL_FEATURES = [
    "device_class",
    "affiliate_id",
    "booker_country",
    "past_city_id",
    "past_hotel_country",
    "year_checkin",
    "month_checkin",
]

class LabelEncoding(Feature):
    def categorical_features(self):
        return []

    def create_features(self):
        train_test_set = load_train_test_set({"input_dir_path": "./data/input/"})
        train_test_set, _, _ = preprocess_train_test_set(train_test_set)
        train_test_set = train_test_set.query("is_last == 1")

        train = train_test_set[train_test_set["row_num"].isnull()].sort_values("utrip_id")
        test = train_test_set[~train_test_set["row_num"].isnull()].sort_values("utrip_id")

        for col in CATEGORICAL_FEATURES:
            self.train_feature[col] = train[col]
            self.test_feature[col] = test[col]

        print(f"train features: {self.train_feature.shape}")
        print(f"test features: {self.test_feature.shape}")


if __name__ == "__main__":
    f = LabelEncoding(path=FE_DIR)
    f.run().save()

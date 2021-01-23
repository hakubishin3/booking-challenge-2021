import sys
import time
import datetime
import contextlib
import pandas as pd
from pathlib import Path
from sklearn import preprocessing


CATEGORICAL_COLS = [
    "booker_country",
    "device_class",
    "affiliate_id",
    "month_checkin",
    "past_hotel_country",
]

def preprocess_train_test_set(train_test_set: pd.DataFrame) -> pd.DataFrame:
    # Shift target values for input sequence.
    unk_city_id = 0
    train_test_set["past_city_id"] = (
        train_test_set.groupby("utrip_id")["city_id"]
        .shift(1)
        .fillna(unk_city_id)
        .astype(int)
    )

    unk_hotel_country = "UNK"
    train_test_set["past_hotel_country"] = (
        train_test_set.groupby("utrip_id")["hotel_country"]
        .shift(1)
        .fillna(unk_hotel_country)
        .astype(str)
    )

    # Encode of target values.
    target_le = preprocessing.LabelEncoder()
    train_test_set["city_id"] = target_le.fit_transform(
        train_test_set["city_id"]
    )
    train_test_set["past_city_id"] = target_le.transform(
        train_test_set["past_city_id"]
    )

    # Change data type
    train_test_set["checkin"] = pd.to_datetime(train_test_set["checkin"])
    train_test_set["checkout"] = pd.to_datetime(train_test_set["checkout"])

    # Add checkin features
    train_test_set["month_checkin"] = train_test_set["checkin"].dt.month
    train_test_set["year_checkin"] = train_test_set["checkin"].dt.year

    # Create days_stay feature.
    train_test_set["days_stay"] = (
        train_test_set["checkout"] - train_test_set["checkin"]
    ).dt.days

    # Create num_checkin feature.
    train_test_set["num_checkin"] = (
        train_test_set.groupby("utrip_id")["checkin"]
        .rank()
    )

    # Create days_move feature.
    train_test_set["past_checkout"] = train_test_set.groupby("utrip_id")[
        "checkout"
    ].shift(1)
    train_test_set["days_move"] = (
        (train_test_set["checkin"] - train_test_set["past_checkout"])
        .dt.days.fillna(0)
    )

    # Encode of categorical values.
    cat_le = {}
    for c in CATEGORICAL_COLS:
        le = preprocessing.LabelEncoder()
        train_test_set[c] = le.fit_transform(
            train_test_set[c].fillna("UNK").astype(str).values
        )
        cat_le[c] = le

    train_test_set["is_last"] = (train_test_set.groupby("utrip_id")["checkin"].rank(ascending=False) == 1).astype(int)

    return train_test_set, target_le, cat_le


def load_train_test_set(config: dict) -> pd.DataFrame:
    train_test_pickle_path = Path(config["input_dir_path"]) / "train_test_set.pickle"
    if train_test_pickle_path.exists():
        train_test_set = pd.read_pickle(train_test_pickle_path)
    else:
        train_set = pd.read_csv(
            Path(config["input_dir_path"]) / "booking_train_set.csv",
            usecols=[
                "user_id",
                "checkin",
                "checkout",
                "city_id",
                "device_class",
                "affiliate_id",
                "booker_country",
                "hotel_country",
                "utrip_id",
            ],
        )
        test_set = pd.read_csv(
            Path(config["input_dir_path"]) / "booking_test_set.csv",
            usecols=[
                "user_id",
                "checkin",
                "checkout",
                "device_class",
                "affiliate_id",
                "booker_country",
                "utrip_id",
                "row_num",  # test only
                "total_rows",  # test only
                "city_id",
                "hotel_country",
            ],
        )
        train_test_set = (
            pd.concat([train_set, test_set], sort=False)
            .sort_values(["utrip_id", "checkin"])
            .reset_index(drop=True)
        )
        train_test_set.to_pickle(train_test_pickle_path)
    return train_test_set


class _Logger:
    def __init__(self, out=sys.stdout):
        self.out = out
        self.fp = None
        self.indent = 0

    def print(self, message: str, *args):
        now = datetime.datetime.now().strftime("%Y/%m/%d %H:%M:%S")
        if len(args) > 0:
            s = f"{now} | {'----' * self.indent}> {message} {' '.join(map(str, args))}"
        else:
            s = f"{now} | {'----' * self.indent}> {message}"
        print(s, file=sys.stdout)
        if self.fp:
            print(s, file=self.fp, flush=True)


_LOGGER = _Logger()


def set_out(f):
    if isinstance(f, (str, Path)):
        f = open(f, "w", encoding="utf-8")
    _LOGGER.fp = f


@contextlib.contextmanager
def span(message: str, *args):
    _LOGGER.print(message, *args)
    start = time.time()
    _LOGGER.indent += 1
    yield
    _LOGGER.indent -= 1
    elapsed = time.time() - start
    _LOGGER.print(f"* {message} ({elapsed:.2f}s)")


def log(message: str, *args):
    _LOGGER.print(message, *args)


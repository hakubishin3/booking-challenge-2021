import sys
import time
import datetime
import contextlib
import pandas as pd
from pathlib import Path


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

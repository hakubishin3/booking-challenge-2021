import yaml
import torch
import argparse
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold

from src import log, set_out, span, load_train_test_set
from src.utils import seed_everything


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def run(config: dict) -> None:
    log("Run with configuration:")
    log(f"{config}")
    seed_everything(config["seed"])

    log("Load train and test set:")
    train_test_set = load_train_test_set(config)
    log(f"{train_test_set.shape}")

    log("Preprocessing:")
    target_le = preprocessing.LabelEncoder()
    train_test_set["city_id"] = target_le.fit_transform(train_test_set["city_id"])

    train = train_test_set[train_test_set["row_num"].isnull()]
    test = train_test_set[~train_test_set["row_num"].isnull()]

    train_trips = train.groupby("utrip_id")["city_id"].apply(list)
    test_trips = test.query("city_id!=0").groupby("utrip_id")["city_id"].apply(list)

    x_train = pd.DataFrame(train_trips)
    x_test = pd.DataFrame(test_trips)

    x_train["n_trips"] = x_train["city_id"].map(lambda x: len(x))
    x_train = x_train.query("n_trips > 2").sort_values("n_trips").reset_index(drop=True)
    x_test = x_test.reset_index(drop=True)
    log(f"x_train: {x_train.shape}, x_test: {x_test.shape}")

    log("Get folds:")
    cv = StratifiedKFold(
        n_splits=config["fold"]["n_splits"],
        shuffle=config["fold"]["shuffle"],
        random_state=config["seed"],
    )
    folds = cv.split(x_train, pd.cut(x_train["n_trips"], 5, labels=False))

    import pdb

    pdb.set_trace()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    output_path = Path(config["output_dir_path"]) / config["exp_name"]
    if not output_path.exists():
        output_path.mkdir(parents=True)
    set_out(output_path / "train_log.txt")

    run(config)


if __name__ == "__main__":
    main()

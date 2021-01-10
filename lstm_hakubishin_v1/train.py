import os
import yaml
import torch
import bisect
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn import preprocessing
from sklearn.model_selection import StratifiedKFold
from torchsummary import summary
from typing import Dict, Type

from src import log, set_out, span, load_train_test_set
from src.utils import seed_everything
from src.dataset import Dataset, Collator
from src.models import BookingLSTM
from src.runner import CustomRunner


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
MODELS: Dict[str, Type[torch.nn.Module]] = {
    cls.__name__: cls
    for cls in [
        BookingLSTM,
    ]
}
CATEGORICAL_COLS = [
    "booker_country",
    "device_class",
    "affiliate_id",
]


def run(config: dict, holdout: bool, debug: bool) -> None:
    log("Run with configuration:")
    log(f"{config}")
    seed_everything(config["seed"])

    with span("Load train and test set:"):
        train_test_set = load_train_test_set(config)
        log(f"{train_test_set.shape}")

    with span("Preprocessing:"):
        log("Shift target values for input sequence.")
        unk_city_id = 0
        train_test_set["past_city_id"] = (
            train_test_set.groupby("utrip_id")["city_id"]
            .shift(1)
            .fillna(unk_city_id)
            .astype(int)
        )

        log("Encode of target values.")
        target_le = preprocessing.LabelEncoder()
        train_test_set["city_id"] = target_le.fit_transform(train_test_set["city_id"])
        train_test_set["past_city_id"] = target_le.transform(
            train_test_set["past_city_id"]
        )

        # 前回のcity_idが0であるレコードを除外する(初回の旅行を除外)
        train_test_set = train_test_set.query("past_city_id != 0").reset_index(
            drop=True
        )

        log("Encode of categorical values.")
        cat_le = {}
        for c in CATEGORICAL_COLS:
            le = preprocessing.LabelEncoder()
            train_test_set[c] = le.fit_transform(
                train_test_set[c].fillna("UNK").astype(str).values
            )
            cat_le[c] = le

        train = train_test_set[train_test_set["row_num"].isnull()]
        test = train_test_set[~train_test_set["row_num"].isnull()]

        x_train, x_test = [], []
        for c in ["city_id", "past_city_id"] + CATEGORICAL_COLS:
            x_train.append(train.groupby("utrip_id")[c].apply(list))
            x_test.append(test.groupby("utrip_id")[c].apply(list))
        x_train = pd.concat(x_train, axis=1)
        x_test = pd.concat(x_test, axis=1)

        x_train["n_trips"] = x_train["city_id"].map(
            lambda x: len(x) + 1
        )  # 前回のcity_idが0であるレコードを除外するので, その分で + 1している
        x_train = (
            x_train.query("n_trips > 2").sort_values("n_trips").reset_index(drop=True)
        )
        x_test = x_test.reset_index(drop=True)
        log(f"x_train: {x_train.shape}, x_test: {x_test.shape}")

        if debug:
            log("'--debug' specified. Shrink data size into 1000.")
            x_train = x_train.iloc[:1000]
            x_test = x_test.iloc[:1000]
            config["params"]["num_epochs"] = 2
            log(f"x_train: {x_train.shape}, x_test: {x_test.shape}")

    with span("Prepare data loader for test:"):
        test_dataset = Dataset(x_test, is_train=False)
        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=config["params"]["bacth_size"],
            num_workers=os.cpu_count(),
            pin_memory=True,
            collate_fn=Collator(is_train=False),
            shuffle=False,
        )

    with span("Get folds:"):
        cv = StratifiedKFold(
            n_splits=config["fold"]["n_splits"],
            shuffle=config["fold"]["shuffle"],
        )
        folds = cv.split(x_train, pd.cut(x_train["n_trips"], 5, labels=False))

    log("Training:")
    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        if holdout and i_fold > 0:
            break
        with span(f"Fold = {i_fold}"):
            x_trn = x_train.loc[trn_idx, :]
            x_val = x_train.loc[val_idx, :]
            train_dataset = Dataset(x_trn, is_train=True)
            valid_dataset = Dataset(x_val, is_train=True)
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=config["params"]["bacth_size"],
                num_workers=os.cpu_count(),
                pin_memory=True,
                collate_fn=Collator(is_train=True),
                shuffle=True,
            )
            valid_dataloader = torch.utils.data.DataLoader(
                valid_dataset,
                batch_size=1,
                num_workers=os.cpu_count(),
                pin_memory=True,
                collate_fn=Collator(is_train=True),
                shuffle=False,
            )
            model_cls = MODELS[config["model_name"]]
            model = model_cls(
                n_city_id=len(target_le.classes_),
                n_booker_country=len(cat_le["booker_country"].classes_),
                n_device_class=len(cat_le["device_class"].classes_),
                n_affiliate_id=len(cat_le["affiliate_id"].classes_),
            )
            if i_fold == 0:
                log(f"{summary(model)}")

            criterion = torch.nn.CrossEntropyLoss()
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=30, eta_min=1e-6
            )
            logdir = (
                Path(config["output_dir_path"]) / config["exp_name"] / f"fold{i_fold}"
            )
            loaders = {"train": train_dataloader, "valid": valid_dataloader}
            runner = CustomRunner(device=DEVICE)
            runner.train(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=scheduler,
                loaders=loaders,
                logdir=logdir,
                num_epochs=config["params"]["num_epochs"],
                verbose=True,
            )

            score = 0
            y_val = x_val["city_id"].map(lambda x: x[-1])
            for loop_i, prediction in enumerate(
                runner.predict_loader(
                    loader=valid_dataloader,
                    resume=f"{logdir}/checkpoints/best.pth",
                    model=model,
                )
            ):
                correct = (
                    y_val.values[loop_i]
                    in np.argsort(prediction.cpu().numpy()[-1, :])[-4:]
                )
                score += int(correct)
            score /= len(y_val)
            log(f"val acc@4: {score}")

            """
            pred = np.array(
                list(
                    map(
                        lambda x: x.cpu().numpy()[-1, :],
                        runner.predict_loader(
                            loader=test_dataloader,
                            resume=f"{logdir}/checkpoints/best.pth",
                            model=model,
                        ),
                    )
                )
            )
            print(pred.shape)
            np.save(Path(config["output_dir_path"]) / config["exp_name"] / f"y_test_pred_fold{i_fold}", pred)
            """


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--holdout", action="store_true")
    args = parser.parse_args()
    with open(args.config, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    output_path = Path(config["output_dir_path"]) / config["exp_name"]
    if not output_path.exists():
        output_path.mkdir(parents=True)
    set_out(output_path / "train_log.txt")

    run(config, args.holdout, args.debug)


if __name__ == "__main__":
    main()

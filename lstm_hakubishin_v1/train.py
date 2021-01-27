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
from transformers import AdamW, get_linear_schedule_with_warmup
from typing import Dict, Type
from sklearn.metrics import top_k_accuracy_score

from src import log, set_out, span, load_train_test_set
from src.utils import seed_everything
from src.dataset import Dataset, Collator
from src.models import BookingLSTM
from src.runner import CustomRunner
from src.losses import FocalLossWithOutOneHot


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
    "month_checkin",
    "past_hotel_country",
]
NUMERICAL_COLS = [
    "days_stay",
    "num_checkin",
    "days_move",
    "num_visit_drop_duplicates",
    "num_visit",
    "num_visit_same_city",
    "num_stay_consecutively",
    "city_embedding",
    "num_past_city_to_embedding",
]


def run(config: dict, holdout: bool, debug: bool) -> None:
    log("Run with configuration:")
    log(f"{config}")
    seed_everything(config["seed"])

    with span("Load train and test set:"):
        train_test_set = load_train_test_set(config)
        log(f"{train_test_set.shape}")
        emb_df = pd.read_csv("./data/interim/emb_df.csv")
        n_emb = emb_df.shape[1] - 1
        emb_cols = [str(i) for i in range(n_emb)]
        emb_df.rename(columns={"city_id": "past_city_id"}, inplace=True)

    with span("Preprocessing:"):
        with span("Shift target values for input sequence."):
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
            train_test_set = pd.merge(train_test_set, emb_df, on="past_city_id", how="left")
            train_test_set[emb_cols] = train_test_set[emb_cols].fillna(0)
            train_test_set["city_embedding"] = train_test_set[emb_cols].apply(lambda x: list(x), axis=1)

        with span("Encode of target values."):
            target_le = preprocessing.LabelEncoder()
            train_test_set["city_id"] = target_le.fit_transform(
                train_test_set["city_id"]
            )
            train_test_set["past_city_id"] = target_le.transform(
                train_test_set["past_city_id"]
            )

        with span("Add features."):
            log("Convert data type of checkin and checkout.")
            train_test_set["checkin"] = pd.to_datetime(train_test_set["checkin"])
            train_test_set["checkout"] = pd.to_datetime(train_test_set["checkout"])

            log("Create month_checkin feature.")
            train_test_set["month_checkin"] = train_test_set["checkin"].dt.month
            train_test_set["year_checkin"] = train_test_set["checkin"].dt.year

            log("Create days_stay feature.")
            train_test_set["days_stay"] = (
                train_test_set["checkout"] - train_test_set["checkin"]
            ).dt.days.apply(lambda x: np.log10(x))

            log("Create num_checkin feature.")
            train_test_set["num_checkin"] = (
                train_test_set.groupby("utrip_id")["checkin"]
                .rank()
                .apply(lambda x: np.log10(x))
            )

            log("Create days_move feature.")
            train_test_set["past_checkout"] = train_test_set.groupby("utrip_id")[
                "checkout"
            ].shift(1)
            train_test_set["days_move"] = (
                (train_test_set["checkin"] - train_test_set["past_checkout"])
                .dt.days.fillna(0)
                .apply(lambda x: np.log1p(x))
            )

            log("Create aggregation features.")
            num_visit_drop_duplicates = train_test_set.query("city_id != 0")[["user_id", "city_id"]].drop_duplicates().groupby("city_id").size().apply(lambda x: np.log1p(x)).reset_index()
            num_visit_drop_duplicates.columns = ["past_city_id", "num_visit_drop_duplicates"]
            num_visit = train_test_set.query("city_id != 0")[["user_id", "city_id"]].groupby("city_id").size().apply(lambda x: np.log1p(x)).reset_index()
            num_visit.columns = ["past_city_id", "num_visit"]
            num_visit_same_city = train_test_set[train_test_set['city_id'] == train_test_set['city_id'].shift(1)].groupby("city_id").size().apply(lambda x: np.log1p(x)).reset_index()
            num_visit_same_city.columns = ["past_city_id", "num_visit_same_city"]
            train_test_set = pd.merge(train_test_set, num_visit_drop_duplicates, on="past_city_id", how="left")
            train_test_set = pd.merge(train_test_set, num_visit, on="past_city_id", how="left")
            train_test_set = pd.merge(train_test_set, num_visit_same_city, on="past_city_id", how="left")
            train_test_set["num_visit_drop_duplicates"].fillna(0, inplace=True)
            train_test_set["num_visit"].fillna(0, inplace=True)
            train_test_set["num_visit_same_city"].fillna(0, inplace=True)
            train_test_set["num_stay_consecutively"] = train_test_set.groupby(["utrip_id", "past_city_id"])["past_city_id"].rank(method="first").fillna(1).apply(lambda x: np.log1p(x))

            log("Create past_city to curent_city features.")
            num_past_to_current_city = train_test_set.groupby(["past_city_id", "city_id"]).count()["user_id"].reset_index()
            num_past_to_current_city.columns = ["past_city_id", "city_id", "num_transition"]
            num_visit_to_past_city = num_past_to_current_city.groupby("past_city_id").sum()["num_transition"].reset_index()
            num_visit_to_past_city.columns = ["past_city_id", "num_past_city_id"]
            num_past_to_current_city = pd.merge(num_past_to_current_city, num_visit_to_past_city, on=["past_city_id"])
            num_past_to_current_city = num_past_to_current_city.set_index(["past_city_id", "city_id"])
            top1000_city_id = train_test_set.groupby("city_id").count().rank(ascending=False).query("user_id <= 1000").index
            transition = pd.DataFrame(train_test_set["past_city_id"].unique(), columns=["past_city_id"])
            past_city_to_cols = []
            transition_idx = num_past_to_current_city.index
            for c_i in top1000_city_id[:100]:
                if c_i == 0:
                    continue
                past_city_to_cols.append("num_past_city_to_{}".format(c_i))
                transition["num_past_city_to_{}".format(c_i)] = transition["past_city_id"].apply(lambda x: np.log1p(num_past_to_current_city.at[(x, c_i), "num_transition"]) if x != 0 and (x, c_i) in transition_idx else 0)
            transition["num_past_city_to_embedding"] = transition[past_city_to_cols].apply(lambda x: list(x), axis=1)
            train_test_set = pd.merge(train_test_set, transition[["past_city_id", "num_past_city_to_embedding"]], on="past_city_id", how="left")

        with span("Encode of categorical values."):
            cat_le = {}
            for c in CATEGORICAL_COLS:
                le = preprocessing.LabelEncoder()
                train_test_set[c] = le.fit_transform(
                    train_test_set[c].fillna("UNK").astype(str).values
                )
                cat_le[c] = le

        train = train_test_set[train_test_set["row_num"].isnull()]
        test = train_test_set[~train_test_set["row_num"].isnull()]

        with span("aggregate features by utrip_id"):
            x_train, x_test_using_train, x_test = [], [], []
            for c in ["city_id", "past_city_id"] + CATEGORICAL_COLS + NUMERICAL_COLS:
                x_train.append(train.groupby("utrip_id")[c].apply(list))
                x_test.append(test.groupby("utrip_id")[c].apply(list))
                x_test_using_train.append(test.groupby("utrip_id")[c].apply(lambda x: list(x)[:-1]))
            x_train = pd.concat(x_train, axis=1)
            x_test = pd.concat(x_test, axis=1)
            x_test_using_train = pd.concat(x_test_using_train, axis=1)

        with span("sampling training data"):
            x_train["n_trips"] = x_train["city_id"].map(lambda x: len(x))
            x_test_using_train["n_trips"] = x_test_using_train["city_id"].map(lambda x: len(x))
            x_train = (
                x_train.query("n_trips > 2")
                .sort_values("n_trips")
                .reset_index(drop=True)
            )
            x_test_using_train = (
                x_test_using_train
                .sort_values("n_trips")
                .reset_index(drop=True)
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
            batch_size=1,
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
    oof_preds = np.zeros((len(x_train), len(target_le.classes_)), dtype=np.float32)
    test_preds = np.zeros((len(x_test), len(target_le.classes_)), dtype=np.float32)

    for i_fold, (trn_idx, val_idx) in enumerate(folds):
        if holdout and i_fold > 0:
            break
        with span(f"Fold = {i_fold}"):
            x_trn = x_train.loc[trn_idx, :]
            x_val = x_train.loc[val_idx, :]
            x_trn = pd.concat([x_trn, x_test_using_train], axis=0, ignore_index=True)
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
                n_month_checkin=len(cat_le["month_checkin"].classes_),
                n_hotel_country=len(cat_le["past_hotel_country"].classes_),
                emb_dim=config["params"]["emb_dim"],
                rnn_dim=config["params"]["rnn_dim"],
                dropout=config["params"]["dropout"],
                rnn_dropout=config["params"]["rnn_dropout"],
            )
            if i_fold == 0:
                log(f"{summary(model)}")

            criterion = FocalLossWithOutOneHot(gamma=0.5)
            # Prepare optimizer
            param_optimizer = list(model.named_parameters())
            no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p
                        for n, p in param_optimizer
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.01,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=1e-4,
                weight_decay=0.01,
            )
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
                main_metric="accuracy04",
                minimize_metric=False,
                logdir=logdir,
                num_epochs=config["params"]["num_epochs"],
                verbose=True,
            )

            log("Predictions using validation data")
            oof_preds[val_idx, :] = np.array(
                list(
                    map(
                        lambda x: x.cpu().numpy()[-1, :],
                        runner.predict_loader(
                            loader=valid_dataloader,
                            resume=f"{logdir}/checkpoints/best.pth",
                            model=model,
                        ),
                    )
                )
            )
            y_val = x_val["city_id"].map(lambda x: x[-1]).values
            score = top_k_accuracy_score(
                y_val, oof_preds[val_idx, :], k=4, labels=np.arange(len(target_le.classes_))
            )
            log(f"val acc@4: {score}")
            np.save(
                Path(config["output_dir_path"])
                / config["exp_name"]
                / f"y_val_pred_fold{i_fold}",
                oof_preds[val_idx, :],
            )

            test_preds_ = np.array(
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
            test_preds += test_preds_ / cv.n_splits
            np.save(
                Path(config["output_dir_path"])
                / config["exp_name"]
                / f"y_test_pred_fold{i_fold}",
                test_preds_,
            )

    log("Evaluation OOF valies:")
    y_train = x_train["city_id"].map(lambda x: x[-1])
    score = top_k_accuracy_score(
        y_train, oof_preds, k=4, labels=np.arange(len(target_le.classes_))
    )
    log(f"oof acc@4: {score}")

    log("Save files:")
    np.save(
        Path(config["output_dir_path"]) / config["exp_name"] / f"y_oof_pred",
        oof_preds,
    )
    np.save(
        Path(config["output_dir_path"]) / config["exp_name"] / f"y_test_pred",
        test_preds,
    )


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

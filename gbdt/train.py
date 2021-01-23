import os
import yaml
import torch
import bisect
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from typing import Dict, Type

from src import log, set_out, span, load_train_test_set, preprocess_train_test_set
from src.utils import seed_everything
from features.base import load_features
from src.models.model import BaseModel
from src.models.lightgbm import ModelLightGBM
from src.runner import Runner


MODELS: Dict[str, Type[BaseModel]] = {
    cls.__name__: cls
    for cls in [
        ModelLightGBM,
    ]
}


def run(config: dict, holdout: bool, debug: bool) -> None:
    log("Run with configuration:")
    log(f"{config}")
    seed_everything(config["seed"])

    with span("Load train and test set:"):
        train_test_set = load_train_test_set(config)
        log(f"{train_test_set.shape}")

    with span("Preprocess train and test set:"):
        train_test_set, target_le, cat_le = preprocess_train_test_set(train_test_set)
        log(f"train_test: {train_test_set.shape}")
        train = train_test_set[train_test_set["row_num"].isnull()]
        test = train_test_set[~train_test_set["row_num"].isnull()]
        log(f"train: {train.shape}")
        log(f"test: {test.shape}")

    with span("Aggregate features by utrip_id"):
        y_train = train.groupby("utrip_id")["city_id"].apply(lambda x: list(x)[-1]).sort_index()
        y_test = test.groupby("utrip_id")["city_id"].apply(lambda x: list(x)[-1]).sort_index()   # All target values are 0.
        log(f"y_train: {y_train.shape}")
        log(f"y_test: {y_test.shape}")

    with span("Get features"):
        x_train, x_test = load_features(config)
        log(f"x_train: {x_train.shape}")
        log(f"x_test: {x_test.shape}")

    with span("Get folds:"):
        cv = StratifiedKFold(
            n_splits=config["fold"]["n_splits"],
            shuffle=config["fold"]["shuffle"],
        )
        folds = cv.split(x_train, pd.cut(x_train["NumericalFeature_num_checkin"], 5, labels=False))
        import pdb; pdb.set_trace()

    with span("Training:"):
        model_cls = MODELS[config["model_name"]]
        model_output_dir = Path(config["output_dir_path"]) / config["exp_name"]
        params = config["params"]
        params["model_params"]["num_class"] = len(target_le.classes_)
        runner = Runner(
            model_cls, params, model_output_dir, f"Train_{model_cls.__name__}", config["fold"]["n_splits"]
        )
        oof_preds, evals_result, importances = runner.train_cv(x_train, y_train, folds)


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
